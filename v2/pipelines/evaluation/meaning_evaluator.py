"""
Evaluation
"""

import re
from typing import Dict, List, Optional

import numpy as np

from v2.graph.fact_verifier import FactVerifier
from v2.logging_config import get_logger
from v2.services.bedrock_service import get_embedding

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(vec1, vec2) -> float:
    v1    = np.array(vec1)
    v2    = np.array(vec2)
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
    executive_summary_text: str,
) -> float:
    """
    Semantic similarity between all section summaries and the executive summary.
    Score: 0-100.
    """
    section_text = " ".join(
        s.get("section_summary", "")
        for s in section_summaries
        if s.get("section_summary")
    )

    if not section_text.strip() or not executive_summary_text.strip():
        logger.warning("Coverage score skipped: empty section or executive summary text")
        return 0.0

    try:
        sec_emb    = get_embedding(section_text)
        exec_emb   = get_embedding(executive_summary_text)
        similarity = _cosine_similarity(sec_emb, exec_emb)
        score      = round(similarity * 100, 2)
        logger.info("Coverage score computed | score=%.2f", score)
        return score
    except Exception:
        logger.exception("Coverage score computation failed")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. FACTUAL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_factual_accuracy(
    section_summaries: List[dict],
    fact_verifier: FactVerifier,
    source_text: str = "",
    max_claims: int = 15,
) -> dict:
    """
    Auto-extracts claims from section summaries and verifies each one
    against the knowledge graph.
    """
    all_claims = []
    for sec in section_summaries:
        text = sec.get("section_summary", "")
        if text:
            all_claims.extend(_extract_sentences(text))

    claims = all_claims[:max_claims]

    if not claims:
        logger.warning("No claims extracted from section summaries")
        return {
            "score": 0.0, "total_claims": 0,
            "supported": 0, "contradicted": 0, "unverified": 0,
            "details": [],
        }

    logger.info("Verifying %d claims for factual accuracy", len(claims))
    results = fact_verifier.verify_batch(claims, source_text)

    supported    = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    unverified   = sum(1 for r in results if r["verdict"] == "UNVERIFIED")

    # Unverified claims do not penalise the score
    denominator = supported + contradicted
    score = round((supported / denominator) * 100, 2) if denominator > 0 else 0.0

    logger.info(
        "Factual accuracy complete | score=%.2f supported=%d contradicted=%d unverified=%d",
        score, supported, contradicted, unverified,
    )

    return {
        "score":        score,
        "total_claims": len(claims),
        "supported":    supported,
        "contradicted": contradicted,
        "unverified":   unverified,
        "details": [
            {
                "claim":      r["claim"],
                "verdict":    r["verdict"],
                "confidence": r["confidence"],
                "reason":     r["reason"],
            }
            for r in results
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTITY EXTRACTION ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

def compute_entity_accuracy(
    extracted_entities: Dict[str, List[str]],
    reference_entities: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    If reference_entities provided  -> precision, recall, F1 per category.
    If no reference                 -> self-consistency score
    """
    if reference_entities:
        logger.info("Entity accuracy mode: reference")
        return _reference_accuracy(extracted_entities, reference_entities)

    logger.info("Entity accuracy mode: self_consistency")
    return _self_consistency_score(extracted_entities)


def _reference_accuracy(
    extracted: Dict[str, List[str]],
    reference: Dict[str, List[str]],
) -> dict:
    """Precision / recall / F1 against a gold-standard reference set."""
    all_categories = set(extracted.keys()) | set(reference.keys())
    per_category   = {}
    total_tp = total_fp = total_fn = 0

    for cat in all_categories:
        ext_set = {_normalize(e) for e in extracted.get(cat, [])}
        ref_set = {_normalize(e) for e in reference.get(cat, [])}

        tp = len(ext_set & ref_set)
        fp = len(ext_set - ref_set)
        fn = len(ref_set - ext_set)

        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1        = (
            round(2 * precision * recall / (precision + recall), 3)
            if (precision + recall) > 0 else 0.0
        )

        per_category[cat] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "extracted": len(ext_set),
            "reference": len(ref_set),
            "matched":   tp,
        }
        logger.debug("Category '%s' | precision=%.3f recall=%.3f f1=%.3f",
                     cat, precision, recall, f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = (
        round(total_tp / (total_tp + total_fp), 3) if (total_tp + total_fp) > 0 else 0.0
    )
    overall_recall = (
        round(total_tp / (total_tp + total_fn), 3) if (total_tp + total_fn) > 0 else 0.0
    )
    overall_f1 = (
        round(2 * overall_precision * overall_recall / (overall_precision + overall_recall), 3)
        if (overall_precision + overall_recall) > 0 else 0.0
    )

    logger.info("Reference accuracy | precision=%.3f recall=%.3f f1=%.3f",
                overall_precision, overall_recall, overall_f1)

    return {
        "mode":    "reference",
        "overall": {
            "precision": overall_precision,
            "recall":    overall_recall,
            "f1":        overall_f1,
        },
        "per_category": per_category,
    }


def _self_consistency_score(extracted: Dict[str, List[str]]) -> dict:
    """
    When no reference is available, score based on:
    - Breadth: how many expected categories have at least one entity
    - Quality: average dedup rate per category (fewer duplicates = better)
    """
    expected_categories = {
        "models", "datasets", "metrics",
        "organizations", "tasks", "key_concepts",
    }

    total_categories  = len(expected_categories)
    filled_categories = sum(1 for cat in expected_categories if extracted.get(cat))
    breadth_score     = round(filled_categories / total_categories, 3)

    per_category = {}
    dedup_rates  = []

    for cat in expected_categories:
        values      = extracted.get(cat, [])
        raw_count   = len(values)
        dedup_count = len({_normalize(v) for v in values})
        dedup_rate  = round(dedup_count / raw_count, 3) if raw_count > 0 else 0.0
        dedup_rates.append(dedup_rate)
        per_category[cat] = {
            "count":      dedup_count,
            "dedup_rate": dedup_rate,
            "entities":   list(dict.fromkeys(values))[:10],
        }
        logger.debug("Category '%s' | count=%d dedup_rate=%.3f", cat, dedup_count, dedup_rate)

    avg_dedup     = round(sum(dedup_rates) / len(dedup_rates), 3) if dedup_rates else 0.0
    overall_score = round((breadth_score + avg_dedup) / 2 * 100, 2)

    logger.info(
        "Self-consistency score | score=%.2f breadth=%.2f avg_dedup=%.2f filled=%d/%d",
        overall_score, breadth_score * 100, avg_dedup * 100,
        filled_categories, total_categories,
    )

    return {
        "mode":    "self_consistency",
        "overall": {
            "score":             overall_score,
            "breadth_score":     round(breadth_score * 100, 2),
            "avg_dedup_rate":    round(avg_dedup * 100, 2),
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
    logger.info("Starting full evaluation")

    logger.info("Step 1/3: coverage score")
    coverage = compute_coverage_score(section_summaries, executive_summary_text)

    logger.info("Step 2/3: factual accuracy")
    factual = compute_factual_accuracy(
        section_summaries, fact_verifier, source_text, max_claims
    )

    logger.info("Step 3/3: entity accuracy")
    entity = compute_entity_accuracy(extracted_entities, reference_entities)

    logger.info("Full evaluation complete | coverage=%.2f factual=%.2f",
                coverage, factual["score"])

    return {
        "coverage": {
            "score":       coverage,
            "description": "Semantic similarity between section summaries and executive summary (0-100)",
        },
        "factual_accuracy": {
            "score":       factual["score"],
            "description": "Percentage of verifiable claims supported by the knowledge graph",
            "details":     factual,
        },
        "entity_accuracy": {
            "score":       entity["overall"].get("score") or entity["overall"].get("f1", 0.0),
            "description": "Entity extraction quality",
            "details":     entity,
        },
    }