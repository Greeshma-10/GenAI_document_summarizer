from typing import List, Dict
from config import settings
from logger import logger
from prompts.executive import (
    build_formatted_input,
    build_research_executive_prompt,
    build_academic_executive_prompt
)
from services.bedrock_service import invoke_llm

# PUBLIC ENTRY=

def generate_executive_summary(
    section_summaries: List[Dict],
    mode: str = "research" 
) -> Dict:

    cleaned = clean_sections(section_summaries)

    if not cleaned:
        return safe_fallback()

    ranked = rank_sections_by_importance(cleaned)
    prioritized = prioritize_sections(ranked)

    if mode == "academic":
        return generate_academic_executive(prioritized)

    return generate_research_executive(prioritized)

# CLEAN SECTIONS

def clean_sections(section_summaries):

    cleaned = []

    for s in section_summaries:
        summary = s.get("section_summary", "").strip()

        if not summary:
            continue

        if "failed" in summary.lower():
            continue

        cleaned.append(s)

    return cleaned

# SECTION RANKING

def rank_sections_by_importance(sections):

    def score(section):

        score = 0
        score += len(section.get("covered_chunk_ids", []))

        summary = section.get("section_summary", "").lower()

        if "architecture" in summary:
            score += 3
        if "framework" in summary:
            score += 2
        if "model" in summary:
            score += 2
        if "method" in summary:
            score += 2

        return score

    return sorted(sections, key=score, reverse=True)

# PRIORITIZATION

def prioritize_sections(ranked_sections):

    if not ranked_sections:
        return []

    top_sections = ranked_sections[:3]
    remaining = ranked_sections[3:]

    compressed = []

    for s in remaining:
        compressed.append({
            "section_id": s["section_id"],
            "section_summary": s["section_summary"],
            "section_key_points": s.get("section_key_points", [])[:3],
            "section_risks_action_items": s.get("section_risks_action_items", [])[:2]
        })

    return top_sections + compressed

# RESEARCH EXECUTIVE GENERATOR

def generate_research_executive(section_summaries: List[Dict]) -> Dict:
    formatted_input = build_formatted_input(section_summaries)
    prompt = build_research_executive_prompt(formatted_input)
    return call_model(prompt)

    
# ACADEMIC EXECUTIVE GENERATOR

def generate_academic_executive(section_summaries: List[Dict]) -> Dict:

    formatted_input = build_formatted_input(section_summaries)
    prompt = build_academic_executive_prompt(formatted_input)
    return call_model(prompt)

# MODEL CALL

def call_model(prompt: str) -> Dict:

    try:
        parsed = invoke_llm(
            prompt,
            max_gen_len=settings.MAX_GEN_LEN_EXEC
        )

        parsed.setdefault("executive_summary", "")
        parsed.setdefault("executive_key_points", [])
        parsed.setdefault("executive_risks_action_items", [])
        parsed.setdefault("tldr", "")

        return parsed

    except Exception as e:
        logger.warning(f"Executive summary generation failed: {str(e)}")
        return safe_fallback()

    return safe_fallback()


# SAFE FALLBACK

def safe_fallback():

    return {
        "executive_summary": "Executive summary generation failed.",
        "executive_key_points": [],
        "executive_risks_action_items": [],
        "tldr": "Executive summary unavailable due to model failure."
    }
