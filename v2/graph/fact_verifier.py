import re
import json
from collections import defaultdict
from typing import List, Dict, Optional

from langchain_ollama import OllamaLLM

from v2.prompts.fact_verification import (
    build_fact_verification_prompt,
    build_claim_extraction_prompt,
)
from v2.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
RELATION_KEYWORDS = {
    "developed_by":  ["developed by", "created by", "built by"],
    "proposed_by":   ["proposed by", "introduced by", "presented by"],
    "uses":          ["uses", "use", "using", "employs", "based on"],
    "trained_on":    ["trained on", "fine-tuned on"],
    "evaluated_on":  ["evaluated on", "tested on", "measured on"],
    "applied_to":    ["applied to", "used for"],
}


class FactVerifier:

    def __init__(self, graph_service, vector_store=None, model_name: str = "llama3"):
        self.graph_service  = graph_service
        self.vector_store   = vector_store
        self.llm            = OllamaLLM(model=model_name)
        logger.info("FactVerifier initialized | vector_store=%s",
                    "Pinecone" if vector_store else "keyword fallback")

    # ─────────────────────────────────────────────────────────────────────────
    def verify(self, claim: str, source_text: str = "") -> dict:
        logger.info("Verifying claim: %s", claim)

        entities       = self._extract_claim_entities(claim)
        logger.debug("Entities extracted: %s", entities)

        graph_evidence = self._get_graph_evidence(entities)
        logger.debug("Graph evidence: %d triples", len(graph_evidence))

        text_evidence  = self._get_text_evidence(claim, source_text)
        logger.debug("Text evidence: %d snippets", len(text_evidence))

        verdict_data   = self._judge_verdict(claim, graph_evidence, text_evidence)

        return {
            "claim":          claim,
            "verdict":        verdict_data.get("verdict", "UNVERIFIED"),
            "reason":         verdict_data.get("reason", ""),
            "confidence":     verdict_data.get("confidence", 0.0),
            "graph_evidence": graph_evidence,
            "text_evidence":  text_evidence,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def verify_batch(self, claims: List[str], source_text: str = "") -> List[dict]:
        return [self.verify(claim, source_text) for claim in claims]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — extract entity names from the claim
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_claim_entities(self, claim: str) -> List[str]:
        # Use a simple inline prompt for claim parsing (not in prompts file —
        # it's a micro-extraction, not a configurable prompt)
        try:
            raw = self.llm.invoke(
                f'Extract subject and object from: "{claim}"\n'
                f'Return ONLY JSON: {{"subject": "...", "object": "..."}}'
            )
            raw    = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
            parsed = json.loads(raw)
            entities = []
            for key in ("subject", "object"):
                val = parsed.get(key, "").strip()
                if val and val.lower() not in ("none", "unknown", ""):
                    entities.append(val)
            return entities
        except Exception:
            logger.debug("Entity extraction fallback for claim: %s", claim)
            words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*', claim)
            return list(dict.fromkeys(words))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — query graph for relevant triples
    # ─────────────────────────────────────────────────────────────────────────
    def _get_graph_evidence(self, entities: List[str]) -> List[dict]:
        evidence = []
        seen     = set()

        for entity in entities:
            try:
                result = self.graph_service.query("neighbours", entity=entity, limit=15)
                rows   = result.get("results", [])

                # Partial word match fallback for multi-word entities
                if not rows and len(entity.split()) > 1:
                    for word in entity.split():
                        if len(word) > 4:
                            try:
                                fallback = self.graph_service.query(
                                    "neighbours", entity=word, limit=10
                                )
                                rows.extend(fallback.get("results", []))
                            except Exception:
                                pass

                for row in rows:
                    key = (row.get("source"), row.get("relation"), row.get("target"))
                    if key not in seen:
                        seen.add(key)
                        evidence.append(row)

            except Exception:
                logger.warning("Graph query failed for entity: %s", entity)

        return evidence

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — find relevant text evidence
    # ─────────────────────────────────────────────────────────────────────────
    def _get_text_evidence(self, claim: str, source_text: str) -> list:
        # Priority 1: Pinecone semantic search
        if self.vector_store:
            try:
                results = self.vector_store.search(claim, top_k=5)
                if results:
                    logger.debug("Pinecone: %d semantic matches", len(results))
                    return results
            except Exception:
                logger.warning("Pinecone search failed, falling back to keyword", exc_info=True)

        # Priority 2: keyword fallback
        if not source_text:
            return []

        keywords  = [w.lower() for w in re.findall(r'\b\w{4,}\b', claim)]
        sentences = re.split(r'(?<=[.!?])\s+', source_text.strip())
        scored    = []

        for sentence in sentences:
            s_lower = sentence.lower()
            score   = sum(1 for kw in keywords if kw in s_lower)
            if score > 0:
                scored.append((score, sentence.strip()))

        scored.sort(reverse=True)
        return [s for _, s in scored[:5]]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — LLM judges verdict
    # ─────────────────────────────────────────────────────────────────────────
    def _judge_verdict(
        self,
        claim: str,
        graph_evidence: List[dict],
        text_evidence: List[str]
    ) -> dict:

        if not graph_evidence and not text_evidence:
            return {
                "verdict":    "UNVERIFIED",
                "reason":     "No evidence found in graph or source text.",
                "confidence": 0.0
            }

        # Pre-LLM rule check — skip LLM for high-confidence rule results
        rule_result = self._rule_based_verdict(claim, graph_evidence)
        if rule_result["verdict"] == "CONTRADICTED" and rule_result["confidence"] >= 0.75:
            logger.debug("Rule-based CONTRADICTED (skipping LLM)")
            return rule_result
        if rule_result["verdict"] == "SUPPORTED" and rule_result["confidence"] >= 0.8:
            logger.debug("Rule-based SUPPORTED (skipping LLM)")
            return rule_result

        # Format evidence for prompt
        graph_str = "\n".join(
            f"  ({r['source']}) -[{r['relation']}]-> ({r['target']})"
            for r in graph_evidence[:10]
        ) if graph_evidence else "  (no graph evidence found)"

        text_str = "\n".join(
            f"  - {s}" for s in text_evidence[:5]
        ) if text_evidence else "  (no text evidence found)"

        # Use centralized prompt — no hardcoding here
        prompt = build_fact_verification_prompt(claim, graph_str, text_str)

        try:
            raw    = self.llm.invoke(prompt)
            raw    = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
            result = json.loads(raw)

            if result.get("verdict") not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIED"):
                result["verdict"] = "UNVERIFIED"

            result["confidence"] = float(result.get("confidence", 0.5))
            return result

        except Exception:
            logger.warning("Verdict LLM failed, falling back to rule-based", exc_info=True)
            return self._rule_based_verdict(claim, graph_evidence)

    # ─────────────────────────────────────────────────────────────────────────
    # Rule-based verdict fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _rule_based_verdict(self, claim: str, graph_evidence: List[dict]) -> dict:
        claim_lower = claim.lower()

        rel_map = defaultdict(list)
        for triple in graph_evidence:
            src = triple.get("source", "").lower()
            rel = triple.get("relation", "").lower()
            tgt = triple.get("target", "").lower()
            rel_map[(src, rel)].append((tgt, triple))

        # Extract claim subject — text before the relation phrase
        claim_subject = None
        for rel_name, phrases in RELATION_KEYWORDS.items():
            for phrase in phrases:
                if phrase in claim_lower:
                    idx         = claim_lower.index(phrase)
                    raw_subject = claim_lower[:idx].strip().rstrip("was is are").strip()
                    for article in ("the ", "a ", "an "):
                        if raw_subject.startswith(article):
                            raw_subject = raw_subject[len(article):]
                    claim_subject = raw_subject.strip()
                    break
            if claim_subject:
                break

        for (src, rel), targets in rel_map.items():

            # Only match triples where source matches claim subject
            if claim_subject:
                if claim_subject not in src and src not in claim_subject:
                    continue
            else:
                if src not in claim_lower:
                    continue

            for tgt, triple in targets:
                if tgt in claim_lower:
                    return {
                        "verdict":    "SUPPORTED",
                        "reason":     (
                            f"Graph confirms: ({triple['source']}) "
                            f"-[{triple['relation']}]-> ({triple['target']})"
                        ),
                        "confidence": 0.8
                    }

            relation_phrases       = RELATION_KEYWORDS.get(rel, [])
            claim_mentions_rel     = any(p in claim_lower for p in relation_phrases)

            if claim_mentions_rel:
                for tgt, triple in targets:
                    tgt_words   = set(tgt.split())
                    claim_words = set(claim_lower.split())
                    if not tgt_words & claim_words:
                        return {
                            "verdict":    "CONTRADICTED",
                            "reason":     (
                                f"Graph shows ({triple['source']}) -[{triple['relation']}]-> "
                                f"({triple['target']}), which conflicts with the claim."
                            ),
                            "confidence": 0.75
                        }

        return {
            "verdict":    "UNVERIFIED",
            "reason":     "Could not match claim to any graph triple.",
            "confidence": 0.0
        }