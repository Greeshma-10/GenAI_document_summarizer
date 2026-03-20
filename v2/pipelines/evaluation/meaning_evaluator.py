"""
Evaluation Module

Computes three metrics required by the project spec:

1. coverage_score        — semantic similarity between section summaries
                           and executive summary (existing meaning_evaluator logic)

2. factual_accuracy      — % of auto-generated claims that are SUPPORTED
                           by the knowledge graph (uses FactVerifier)

3. entity_accuracy       — precision / recall / F1 of extracted entities
                           compared to a reference set (if provided),
                           or a self-consistency score (if no reference)
"""

import re
from typing import List, Dict, Optional
import numpy as np
from v1.services.bedrock_service import get_embedding
from v2.graph.fact_verifier import FactVerifier


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(vec1, vec2) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def _extract_sentences(text: str) -> List[str]:
    """Split text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _normalize(entity: str) -> str:
    return entity.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. COVERAGE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage_score(
    section_summaries: List[dict],
    executive_summary_text: str
) -> float:
    """
    Semantic similarity between all section summaries and the executive summary.
    Score: 0–100
    """
    section_text = " ".join(
        s.get("section_summary", "")
        for s in section_summaries
        if s.get("section_summary")
    )

    if not section_text.strip() or not executive_summary_text.strip():
        return 0.0

    try:
        sec_emb  = get_embedding(section_text)
        exec_emb = get_embedding(executive_summary_text)
        similarity = _cosine_similarity(sec_emb, exec_emb)
        return round(similarity * 100, 2)
    except Exception as e:
        print(f"⚠️ Coverage score failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACTUAL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_factual_accuracy(
    section_summaries: List[dict],
    fact_verifier: FactVerifier,
    source_text: str = "",
    max_claims: int = 15
) -> dict:
    """
    Auto-extracts claims from section summaries and verifies each one
    against the knowledge graph.

    Returns:
      {
        "score":         float  (0–100, % of claims supported),
        "total_claims":  int,
        "supported":     int,
        "contradicted":  int,
        "unverified":    int,
        "details":       list[dict]
      }
    """
    # Collect sentences from all section summaries as claims
    all_claims = []
    for sec in section_summaries:
        text = sec.get("section_summary", "")
        if text:
            all_claims.extend(_extract_sentences(text))

    # Cap to avoid very long runs
    claims = all_claims[:max_claims]

    if not claims:
        return {
            "score": 0.0, "total_claims": 0,
            "supported": 0, "contradicted": 0, "unverified": 0,
            "details": []
        }

    print(f"📊 Verifying {len(claims)} claims for factual accuracy...")
    results = fact_verifier.verify_batch(claims, source_text)

    supported    = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    unverified   = sum(1 for r in results if r["verdict"] == "UNVERIFIED")

    # Score = supported / (supported + contradicted)
    # Unverified claims don't penalise the score
    denominator = supported + contradicted
    score = round((supported / denominator) * 100, 2) if denominator > 0 else 0.0

    return {
        "score":        score,
        "total_claims": len(claims),
        "supported":    supported,
        "contradicted": contradicted,
        "unverified":   unverified,
        "details":      [
            {
                "claim":      r["claim"],
                "verdict":    r["verdict"],
                "confidence": r["confidence"],
                "reason":     r["reason"],
            }
            for r in results
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTITY EXTRACTION ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_entity_accuracy(
    extracted_entities: Dict[str, List[str]],
    reference_entities: Optional[Dict[str, List[str]]] = None
) -> dict:
    """
    If reference_entities provided → compute precision, recall, F1 per category.
    If no reference              → compute self-consistency score
                                   (ratio of non-empty categories, dedup rate).

    Returns:
      {
        "mode":       "reference" | "self_consistency",
        "overall":    { "precision", "recall", "f1" }  or  { "score" },
        "per_category": { category: { ... } }
      }
    """

    if reference_entities:
        return _reference_accuracy(extracted_entities, reference_entities)
    else:
        return _self_consistency_score(extracted_entities)


def _reference_accuracy(
    extracted: Dict[str, List[str]],
    reference: Dict[str, List[str]]
) -> dict:
    """Precision / recall / F1 against a gold standard reference set."""

    all_categories = set(extracted.keys()) | set(reference.keys())
    per_category = {}
    total_tp = total_fp = total_fn = 0

    for cat in all_categories:
        ext_set = set(_normalize(e) for e in extracted.get(cat, []))
        ref_set = set(_normalize(e) for e in reference.get(cat, []))

        tp = len(ext_set & ref_set)
        fp = len(ext_set - ref_set)
        fn = len(ref_set - ext_set)

        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1        = round(
            2 * precision * recall / (precision + recall), 3
        ) if (precision + recall) > 0 else 0.0

        per_category[cat] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "extracted": len(ext_set),
            "reference": len(ref_set),
            "matched":   tp,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = round(total_tp / (total_tp + total_fp), 3) if (total_tp + total_fp) > 0 else 0.0
    overall_recall    = round(total_tp / (total_tp + total_fn), 3) if (total_tp + total_fn) > 0 else 0.0
    overall_f1        = round(
        2 * overall_precision * overall_recall / (overall_precision + overall_recall), 3
    ) if (overall_precision + overall_recall) > 0 else 0.0

    return {
        "mode":         "reference",
        "overall":      {
            "precision": overall_precision,
            "recall":    overall_recall,
            "f1":        overall_f1,
        },
        "per_category": per_category,
    }


def _self_consistency_score(extracted: Dict[str, List[str]]) -> dict:
    """
    When no reference is available, score based on:
    - How many categories have at least one entity (breadth)
    - Average dedup rate per category (quality — fewer duplicates = better)
    """
    expected_categories = {
        "models", "datasets", "metrics",
        "organizations", "tasks", "key_concepts"
    }

    total_categories   = len(expected_categories)
    filled_categories  = sum(
        1 for cat in expected_categories
        if extracted.get(cat)
    )
    breadth_score = round(filled_categories / total_categories, 3)

    per_category = {}
    dedup_rates  = []

    for cat in expected_categories:
        values = extracted.get(cat, [])
        raw_count  = len(values)
        dedup_count = len(set(_normalize(v) for v in values))
        dedup_rate  = round(dedup_count / raw_count, 3) if raw_count > 0 else 0.0
        dedup_rates.append(dedup_rate)
        per_category[cat] = {
            "count":      dedup_count,
            "dedup_rate": dedup_rate,
            "entities":   list(dict.fromkeys(values))[:10],  # show first 10
        }

    avg_dedup = round(sum(dedup_rates) / len(dedup_rates), 3) if dedup_rates else 0.0
    overall_score = round((breadth_score + avg_dedup) / 2 * 100, 2)

    return {
        "mode":    "self_consistency",
        "overall": {
            "score":          overall_score,
            "breadth_score":  round(breadth_score * 100, 2),
            "avg_dedup_rate": round(avg_dedup * 100, 2),
            "filled_categories": f"{filled_categories}/{total_categories}",
        },
        "per_category": per_category,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EVALUATION — runs all three
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    section_summaries: List[dict],
    executive_summary_text: str,
    extracted_entities: Dict[str, List[str]],
    fact_verifier: FactVerifier,
    source_text: str = "",
    reference_entities: Optional[Dict[str, List[str]]] = None,
    max_claims: int = 15,
) -> dict:
    """
    Runs all three evaluations and returns a combined report.
    """
    print("\n📊 Running full evaluation...")

    print("  1/3 Coverage score...")
    coverage = compute_coverage_score(section_summaries, executive_summary_text)

    print("  2/3 Factual accuracy...")
    factual = compute_factual_accuracy(
        section_summaries, fact_verifier, source_text, max_claims
    )

    print("  3/3 Entity accuracy...")
    entity = compute_entity_accuracy(extracted_entities, reference_entities)

    print("  ✅ Evaluation complete")

    return {
        "coverage": {
            "score":       coverage,
            "description": "Semantic similarity between section summaries and executive summary (0–100)"
        },
        "factual_accuracy": {
            "score":       factual["score"],
            "description": "% of verifiable claims supported by the knowledge graph",
            "details":     factual,
        },
        "entity_accuracy": {
            "score":       factual["score"],   # use factual as proxy if no reference
            "description": "Entity extraction quality",
            "details":     entity,
        },
    }