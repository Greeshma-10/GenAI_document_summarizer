"""
Fact Verification Module

Given a claim (string), verifies it against:
  1. The Neo4j knowledge graph (structured evidence)
  2. The source text via keyword search (unstructured evidence)

Returns:
  verdict:  SUPPORTED | CONTRADICTED | UNVERIFIED
  evidence: list of supporting/contradicting triples or text snippets
  score:    0.0 - 1.0 confidence
"""

import re
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# Prompt: extract a structured claim from free text
# ─────────────────────────────────────────────────────────────────────────────
CLAIM_PARSE_PROMPT = PromptTemplate(
    input_variables=["claim"],
    template="""
Extract the subject, relation, and object from this claim.

Claim: {claim}

Return ONLY valid JSON — no explanation, no markdown:
{{"subject": "...", "relation": "...", "object": "..."}}
"""
)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt: judge verdict given evidence
# ─────────────────────────────────────────────────────────────────────────────
VERDICT_PROMPT = PromptTemplate(
    input_variables=["claim", "graph_evidence", "text_evidence"],
    template="""
You are a strict fact-checking system. Analyse the claim against the evidence and return a verdict.

Claim: {claim}

Graph evidence (structured knowledge graph triples):
{graph_evidence}

Text evidence (sentences from source document):
{text_evidence}

Verdict rules — follow these exactly:

CONTRADICTED — use this when:
  - The claim names entity X but the graph shows a DIFFERENT entity Y for the same relation
    e.g. claim says "developed by OpenAI" but graph shows DEVELOPED_BY -> Google Brain → CONTRADICTED
  - The claim says the subject USES something but the graph shows it uses something completely different
    e.g. claim says "uses convolutional layers" but graph shows USES -> self-attention → CONTRADICTED
  - The claim names an organization or person that does NOT appear anywhere in the evidence
    but a DIFFERENT organization/person appears for the same role → CONTRADICTED

SUPPORTED — use this when:
  - The graph contains a triple that directly matches the claim's subject, relation, and object
  - e.g. claim "Transformer uses self-attention" and graph has (Transformer)-[USES]->(self-attention) → SUPPORTED

UNVERIFIED — use this ONLY when:
  - There is genuinely no relevant evidence at all (graph and text are both empty or unrelated)
  - Do NOT use UNVERIFIED when contradicting evidence exists

Examples:
  Claim: "Transformer was developed by OpenAI"
  Graph: (Transformer)-[DEVELOPED_BY]->(Google Brain)
  → CONTRADICTED — graph shows Google Brain, not OpenAI

  Claim: "Transformer uses recurrent neural networks"
  Graph: (Transformer)-[USES]->(self-attention), (Transformer)-[USES]->(Multi-Head Attention)
  → CONTRADICTED — graph shows attention mechanisms, not recurrent networks

  Claim: "Transformer was developed by Google Brain"
  Graph: (Transformer)-[DEVELOPED_BY]->(Google Brain)
  → SUPPORTED

Return ONLY valid JSON — no explanation, no markdown, no backticks:
{{
  "verdict": "SUPPORTED",
  "reason": "one sentence explanation referencing specific evidence",
  "confidence": 0.0
}}
"""
)


class FactVerifier:

    def __init__(self,graph_service,vector_store=None,model_name: str = "llama3"):
        self.graph_service = graph_service
        self.vector_store   = vector_store
        self.llm = OllamaLLM(model=model_name)
        self.claim_prompt  = CLAIM_PARSE_PROMPT
        self.verdict_prompt = VERDICT_PROMPT

    # ─────────────────────────────────────────────────────────────────────────
    def verify(self, claim: str, source_text: str = "") -> dict:
        """
        Main entry point.

        Args:
            claim:       the statement to verify e.g. "Transformer was developed by Google Brain"
            source_text: original document text for unstructured search (optional)

        Returns:
            {
              "claim":          str,
              "verdict":        "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED",
              "reason":         str,
              "confidence":     float,
              "graph_evidence": list[dict],
              "text_evidence":  list[str],
            }
        """
        print(f"\n🔍 VERIFYING: {claim}")

        # Step 1: extract entities from the claim for graph lookup
        entities = self._extract_claim_entities(claim)
        print(f"   Entities extracted: {entities}")

        # Step 2: query graph for evidence
        graph_evidence = self._get_graph_evidence(entities)
        print(f"   Graph evidence: {len(graph_evidence)} triples")

        # Step 3: search source text for evidence
        text_evidence = self._get_text_evidence(claim, source_text)
        print(f"   Text evidence: {len(text_evidence)} snippets")

        # Step 4: LLM judges the verdict
        verdict_data = self._judge_verdict(claim, graph_evidence, text_evidence)

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
        """Verify multiple claims at once."""
        return [self.verify(claim, source_text) for claim in claims]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — extract key entity names from the claim
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_claim_entities(self, claim: str) -> List[str]:
        """
        Extract subject + object entity names from the claim
        using the LLM, then fall back to simple noun extraction.
        """
        try:
            raw = self.llm.invoke(self.claim_prompt.format(claim=claim))
            raw = re.sub(r"```(?:json)?|```", "", str(raw)).strip()
            import json
            parsed = json.loads(raw)
            entities = []
            for key in ("subject", "object"):
                val = parsed.get(key, "").strip()
                if val and val.lower() not in ("none", "unknown", ""):
                    entities.append(val)
            return entities
        except Exception:
            # Fallback: split claim into words, take capitalized ones
            words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*', claim)
            return list(dict.fromkeys(words))  # dedup preserving order

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — query the graph for relevant triples
    # ─────────────────────────────────────────────────────────────────────────
    def _get_graph_evidence(self, entities: List[str]) -> List[dict]:
        """
        Get all triples involving any of the extracted entities.
        Also tries partial/keyword matches when exact entity lookup returns nothing.
        """
        evidence = []
        seen = set()

        for entity in entities:
            try:
                result = self.graph_service.query(
                    "neighbours", entity=entity, limit=15
                )
                rows = result.get("results", [])

                # If exact match found nothing, try each word in entity name
                # e.g. "recurrent neural networks" → try "recurrent"
                if not rows and len(entity.split()) > 1:
                    for word in entity.split():
                        if len(word) > 4:   # skip short words
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

            except Exception as e:
                print(f"   ⚠️ Graph query failed for '{entity}': {e}")

        return evidence

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — find relevant sentences in source text
    # ─────────────────────────────────────────────────────────────────────────
    def _get_text_evidence(self, claim: str, source_text: str) -> list:
        """
        Find relevant text evidence for a claim.
 
        Priority:
          1. Pinecone semantic search (if vector_store available)
          2. Keyword fallback (if no vector store or search fails)
        """
        # ── Option 1: Pinecone semantic search ───────────────────────────────
        if self.vector_store:
            try:
                results = self.vector_store.search(claim, top_k=5)
                if results:
                    print(f"   🔍 Pinecone: {len(results)} semantic matches")
                    return results
            except Exception as e:
                print(f"   ⚠️ Pinecone search failed, falling back to keyword: {e}")
 
        # ── Option 2: keyword fallback ────────────────────────────────────────
        if not source_text:
            return []
 
        import re
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
        """Ask the LLM to judge SUPPORTED / CONTRADICTED / UNVERIFIED."""

        # Format evidence for prompt
        if graph_evidence:
            graph_str = "\n".join(
                f"  ({r['source']}) -[{r['relation']}]-> ({r['target']})"
                for r in graph_evidence[:10]
            )
        else:
            graph_str = "  (no graph evidence found)"

        text_str = "\n".join(
            f"  - {s}" for s in text_evidence[:5]
        ) if text_evidence else "  (no text evidence found)"

        # If no evidence at all, skip LLM call
        if not graph_evidence and not text_evidence:
            return {
                "verdict": "UNVERIFIED",
                "reason": "No evidence found in graph or source text.",
                "confidence": 0.0
            }

        # ── Pre-LLM rule check ────────────────────────────────────────────
        # Run rule-based check first — if it gives a high-confidence answer
        # trust it and skip the LLM (avoids hallucination on clear cases)
        rule_result = self._rule_based_verdict(claim, graph_evidence)
        if rule_result["verdict"] == "CONTRADICTED" and rule_result["confidence"] >= 0.75:
            print(f"   ⚡ Rule-based CONTRADICTED (skipping LLM)")
            return rule_result
        if rule_result["verdict"] == "SUPPORTED" and rule_result["confidence"] >= 0.8:
            print(f"   ⚡ Rule-based SUPPORTED (skipping LLM)")
            return rule_result

        try:
            raw = self.llm.invoke(self.verdict_prompt.format(
                claim=claim,
                graph_evidence=graph_str,
                text_evidence=text_str
            ))
            raw = re.sub(r"```(?:json)?|```", "", str(raw)).strip()

            import json
            result = json.loads(raw)

            # Validate verdict value
            if result.get("verdict") not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIED"):
                result["verdict"] = "UNVERIFIED"

            result["confidence"] = float(result.get("confidence", 0.5))
            return result

        except Exception as e:
            print(f"   ⚠️ Verdict LLM failed: {e}")
            # Rule-based fallback: if graph has matching triple, mark supported
            return self._rule_based_verdict(claim, graph_evidence)

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback: rule-based verdict from graph triples alone
    # ─────────────────────────────────────────────────────────────────────────
    def _rule_based_verdict(self, claim: str, graph_evidence: List[dict]) -> dict:
        """
        Rule-based verdict using keyword matching.

        SUPPORTED    — subject AND object both appear in claim AND a graph triple
        CONTRADICTED — subject appears in claim, graph has a triple with the SAME
                       relation keyword but a DIFFERENT object
        UNVERIFIED   — no relevant match found
        """
        claim_lower = claim.lower()

        # Relation keyword → canonical relation name mapping
        RELATION_KEYWORDS = {
            "developed_by":  ["developed by", "created by", "built by"],
            "proposed_by":   ["proposed by", "introduced by", "presented by"],
            "uses":          ["uses", "use", "using", "employs", "based on"],
            "trained_on":    ["trained on", "fine-tuned on"],
            "evaluated_on":  ["evaluated on", "tested on", "measured on"],
            "applied_to":    ["applied to", "used for"],
        }

        from collections import defaultdict
        # Group graph triples by (normalised_source, relation)
        rel_map = defaultdict(list)
        for triple in graph_evidence:
            src = triple.get("source", "").lower()
            rel = triple.get("relation", "").lower()
            tgt = triple.get("target", "").lower()
            rel_map[(src, rel)].append((tgt, triple))

        # Extract the claim subject — the first noun phrase before the relation verb
        # e.g. "BLEU was developed by Google Brain" → subject is "bleu"
        # e.g. "The Transformer uses RNNs" → subject is "transformer"
        claim_subject = None
        for rel_name, phrases in RELATION_KEYWORDS.items():
            for phrase in phrases:
                if phrase in claim_lower:
                    # Subject is everything before the relation phrase
                    idx = claim_lower.index(phrase)
                    raw_subject = claim_lower[:idx].strip().rstrip("was is are").strip()
                    # Remove leading articles
                    for article in ("the ", "a ", "an "):
                        if raw_subject.startswith(article):
                            raw_subject = raw_subject[len(article):]
                    claim_subject = raw_subject.strip()
                    break
            if claim_subject:
                break

        for (src, rel), targets in rel_map.items():

            # Only match triples whose source is the SUBJECT of the claim
            # not just any entity mentioned anywhere in the claim
            if claim_subject:
                if claim_subject not in src and src not in claim_subject:
                    continue
            else:
                if src not in claim_lower:
                    continue

            for tgt, triple in targets:
                # SUPPORTED — object also appears in claim
                if tgt in claim_lower:
                    return {
                        "verdict": "SUPPORTED",
                        "reason": (
                            f"Graph confirms: ({triple['source']}) "
                            f"-[{triple['relation']}]-> ({triple['target']})"
                        ),
                        "confidence": 0.8
                    }

            # CONTRADICTED — only fire if:
            # 1. claim contains a keyword for THIS specific relation
            # 2. graph object is NOT in the claim
            relation_phrases = RELATION_KEYWORDS.get(rel, [])
            claim_mentions_this_relation = any(
                phrase in claim_lower for phrase in relation_phrases
            )

            if claim_mentions_this_relation:
                for tgt, triple in targets:
                    tgt_words = set(tgt.split())
                    claim_words = set(claim_lower.split())
                    if not tgt_words & claim_words:
                        return {
                            "verdict": "CONTRADICTED",
                            "reason": (
                                f"Graph shows ({triple['source']}) -[{triple['relation']}]-> "
                                f"({triple['target']}), which conflicts with the claim."
                            ),
                            "confidence": 0.75
                        }

        return {
            "verdict": "UNVERIFIED",
            "reason": "Could not match claim to any graph triple.",
            "confidence": 0.0
        }