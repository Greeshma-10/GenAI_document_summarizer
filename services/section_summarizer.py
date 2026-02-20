from typing import List, Dict
from config import settings
from logger import logger
from prompts.section import build_section_prompt
from services.bedrock_service import invoke_llm

# PUBLIC 

def summarize_section(section_chunks: List[Dict], section_id: int) -> Dict:

    if len(section_chunks) == 1:
        chunk = section_chunks[0]

        return {
            "section_id": section_id,
            "section_summary": chunk.get("summary", ""),
            "section_key_points": chunk.get("key_points", []),
            "section_risks_action_items": chunk.get("key_risks_action_items", []),
            "covered_chunk_ids": [chunk.get("chunk_id")]
        }

    return summarize_section_bedrock(section_chunks, section_id)


# MAIN ENGINE

def summarize_section_bedrock(section_chunks: List[Dict], section_id: int) -> Dict:

    prompt = build_section_prompt(section_chunks, section_id)

    try:
        parsed = invoke_llm(
            prompt,
            max_gen_len=settings.MAX_GEN_LEN_SECTION
        )

        parsed.setdefault("section_id", section_id)
        parsed.setdefault("section_summary", "")
        parsed.setdefault("section_key_points", [])
        parsed.setdefault("section_risks_action_items", [])

        if not isinstance(parsed["section_key_points"], list):
            parsed["section_key_points"] = []

        if not isinstance(parsed["section_risks_action_items"], list):
            parsed["section_risks_action_items"] = []

        parsed["covered_chunk_ids"] = [
            c["chunk_id"] for c in section_chunks
        ]

        return parsed

    except Exception as e:
        logger.warning(f"Section {section_id} summarization failed: {str(e)}")

    # FINAL SAFE FALLBACK

    combined_summary = " ".join(
        c.get("summary", "") for c in section_chunks
    )[:500]

    return {
        "section_id": section_id,
        "section_summary": combined_summary,
        "section_key_points": [],
        "section_risks_action_items": [],
        "covered_chunk_ids": [c["chunk_id"] for c in section_chunks]
    }
