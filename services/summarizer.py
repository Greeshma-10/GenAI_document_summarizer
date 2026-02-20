import json
import time
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings
from logger import logger
from prompts.chunk import build_chunk_summary_prompt
from services.bedrock_service import invoke_llm

MAX_WORKERS = settings.MAX_WORKERS


# LOW INFORMATION DETECTOR 

def is_low_information_chunk(text: str) -> bool:

    if not text or not text.strip():
        return True

    # Table override
    if looks_like_table(text):
        return False

    if len(text.split()) < 40:
        return True

    if re.search(r"\[\d+\]", text) and len(text.split()) < 80:
        return True

    lines = text.splitlines()
    short_lines = sum(1 for l in lines if len(l.split()) < 5)

    if short_lines > len(lines) * 0.65:
        return True

    return False

# GROUNDEDNESS CHECK

def is_grounded(summary: str, source_text: str, mode: str = "research") -> bool:

    if not summary.strip():
        return False

    summary_lower = summary.lower()
    source_lower = source_text.lower()

    # -------- Phrase grounding --------
    summary_words = summary_lower.split()

    phrases = [
        " ".join(summary_words[i:i+3])
        for i in range(len(summary_words) - 2)
    ]

    phrase_matches = [
        phrase for phrase in phrases
        if phrase in source_lower
    ]

    phrase_ratio = len(phrase_matches) / len(phrases) if phrases else 0

    # -------- Keyword grounding (regex-safe) --------
    clean_words = re.findall(r"[a-zA-Z]+", summary_lower)

    keywords = [
        word for word in clean_words
        if len(word) > 6
    ]

    keyword_matches = [
        word for word in keywords
        if word in source_lower
    ]

    keyword_ratio = len(keyword_matches) / len(keywords) if keywords else 0

    if mode == "research":
        return phrase_ratio >= 0.08

    if mode == "academic":
        phrase_condition = phrase_ratio >= 0.04
        keyword_condition = keyword_ratio >= 0.30 and len(keyword_matches) >= 3
        return phrase_condition or keyword_condition

    return False

# CLEAN + DEDUPLICATION

def clean_string_list(items):

    if not isinstance(items, list):
        return []

    return [
        item.strip()
        for item in items
        if isinstance(item, str) and item.strip()
    ]

def deduplicate(items):

    seen = set()
    result = []

    for item in items:
        normalized = item.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(item)

    return result

def looks_like_table(text: str) -> bool:

    lines = text.splitlines()

    if len(lines) < 3:
        return False

    numeric_lines = 0
    structured_lines = 0

    for line in lines:
        tokens = line.strip().split()

        # Many columns
        if len(tokens) >= 3:
            structured_lines += 1

        # Contains numbers or percentages
        if re.search(r"\d", line):
            numeric_lines += 1

    # If many structured rows AND numeric content → table
    if structured_lines >= len(lines) * 0.6 and numeric_lines >= 2:
        return True

    return False

def _process_single_chunk(idx, chunk, total_chunks, mode):

    logger.info(f"Processing chunk {idx}/{total_chunks} (mode={mode})")

    try:

        if is_low_information_chunk(chunk):
            if mode == "research":
                logger.info(f"Chunk {idx} identified as low-information. Discarding.")
                return {
                    "chunk_id": idx,
                    "summary": "",
                    "key_points": [],
                    "key_risks_action_items": []
                }
            else:
                logger.info(f"Chunk {idx} identified as low-information. Keeping (academic mode).")

        parsed = invoke_llm(
        build_chunk_summary_prompt(chunk, idx),
        max_gen_len=settings.MAX_GEN_LEN_CHUNK
        )

        summary = parsed.get("summary", "").strip()

        if not is_grounded(summary, chunk, mode=mode):

            if mode == "research":
                logger.warning(f"Chunk {idx} summary failed groundedness check. Discarding.")
                return {
                    "chunk_id": idx,
                    "summary": "",
                    "key_points": [],
                    "key_risks_action_items": []
                }
            else:
                logger.warning(f"Chunk {idx} summary failed groundedness check. Keeping (academic mode).")
                
        cleaned_key_points = clean_string_list(parsed.get("key_points", []))
        cleaned_risks = clean_string_list(parsed.get("key_risks_action_items", []))

        cleaned_key_points = deduplicate(cleaned_key_points)
        cleaned_risks = deduplicate(cleaned_risks)

        return {
            "chunk_id": idx,
            "summary": summary,
            "key_points": cleaned_key_points,
            "key_risks_action_items": cleaned_risks
        }

    except Exception as e:
        logger.warning(f"Chunk {idx} processing failed: {str(e)}")

        return {
            "chunk_id": idx,
            "summary": "",
            "key_points": [],
            "key_risks_action_items": []
        }

# PUBLIC SUMMARIZER

def summarize_chunks(chunks: List[str], mode: str = "academic") -> List[Dict]:

    total_chunks = len(chunks)
    print(f"\nProcessing {total_chunks} chunks (mode={mode})\n")

    results = [None] * total_chunks
    MAX_WORKERS = settings.MAX_WORKERS  

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(_process_single_chunk, idx, chunk, total_chunks, mode)
            for idx, chunk in enumerate(chunks, start=1)
        ]

        for future in as_completed(futures):
            result = future.result()
            results[result["chunk_id"] - 1] = result

    return results
