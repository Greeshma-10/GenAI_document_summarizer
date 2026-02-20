from schema.document_schema import FinalOutput
from typing import List, Dict

# SECTION FAILURE DETECTOR

def is_section_invalid(section: Dict) -> bool:

    summary = section.get("section_summary", "")

    if not summary:
        return True

    summary_lower = summary.lower()

    return (
        "failed" in summary_lower or
        "invalid json" in summary_lower or
        summary_lower.strip() == ""
    )

# MEANINGFUL CHUNK DETECTOR

def is_chunk_meaningful(chunk: Dict) -> bool:

    if chunk.get("model_error", False):
        return False

    return (
        chunk.get("summary", "").strip() != ""
        or bool(chunk.get("key_points"))
        or any(item.strip() for item in chunk.get("key_risks_action_items", []) if isinstance(item, str))

    )

# COVERAGE CALCULATION (SEMANTIC-AWARE)

def calculate_coverage(section_outputs, chunk_outputs):

    total_chunks = len(chunk_outputs)

    if total_chunks == 0:
        return 0.0

    meaningful_chunks = {
        c.get("chunk_id")
        for c in chunk_outputs
        if is_chunk_meaningful(c)
    }

    if not meaningful_chunks:
        return 0.0

    covered = set()

    for section in section_outputs:

        if is_section_invalid(section):
            continue

        ids = section.get("covered_chunk_ids", [])

        if isinstance(ids, list):
            for i in ids:
                if i in meaningful_chunks:
                    covered.add(i)

    if len(meaningful_chunks) == 0:
        return 0.0

    coverage_score = (len(covered) / len(meaningful_chunks)) * 100


    return round(coverage_score, 2)

# SECTION SEMANTIC EMPTY CHECK

def is_section_semantically_empty(section: Dict) -> bool:

    return not (
        section.get("section_summary", "").strip()
        or section.get("section_key_points")
        or section.get("section_risks_action_items")
    )

# FINAL DOCUMENT ASSEMBLY

def assemble_document(
        executive_output: Dict,
        section_outputs: List[Dict],
        chunk_outputs: List[Dict],
        total_chunks: int,
):

    # Executive Safe Defaults

    executive_summary = executive_output.get("executive_summary", "")
    executive_key_points = executive_output.get("executive_key_points", [])
    executive_risks = executive_output.get("executive_risks_action_items", [])
    tldr = executive_output.get("tldr", "Executive TLDR unavailable.")

    if not isinstance(executive_key_points, list):
        executive_key_points = []

    if not isinstance(executive_risks, list):
        executive_risks = []

    # Coverage Calculation

    coverage_score = calculate_coverage(section_outputs, chunk_outputs)

    # GLOBAL MEANINGFUL CONTENT CHECK 

    meaningful_chunks = [
        c for c in chunk_outputs
        if is_chunk_meaningful(c)
    ]

    missing_section_flag = False

    # If no grounded/meaningful chunks exist → document invalid
    if not meaningful_chunks:
        missing_section_flag = True

    # Section-Level Validation

    for section in section_outputs:

        if is_section_invalid(section):
            missing_section_flag = True
            break

        if is_section_semantically_empty(section):
            missing_section_flag = True
            break

    # Executive summary missing check
    if not executive_summary.strip():
        missing_section_flag = True

    # Coverage Flag Logic

    coverage_flag = coverage_score < 70 or missing_section_flag

    document = {
        "document_summary": {
            "tldr": tldr,
            "executive_summary": executive_summary,
            "key_points": executive_key_points,
            "risks_action_items": executive_risks,
            "sections": section_outputs,
            "chunk_summaries": chunk_outputs,
            "coverage_score": coverage_score,
            "coverage_flag": coverage_flag,
            "missing_section_flag": missing_section_flag
        }
    }

    return FinalOutput(**document)
