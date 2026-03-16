from fastapi import FastAPI, UploadFile, File, Form
from v2.ingestion.document_parser import parse_document
from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.document_assembler import assemble_document
from v2.pipelines.evaluation.meaning_evaluator import compute_meaning_coverage
from v2.pipelines.entity_extraction.entity_extractor import extract_entities

import time
import tempfile
import os

app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "API is healthy and running!"}


@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    mode: str = Form("academic")
):

    if mode not in ["academic", "research"]:
        mode = "academic"

    document_mode = mode

    total_start = time.time()

    # INGESTION
    ingestion_start = time.time()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    document_data = parse_document(temp_path)

    ingestion_time = round(time.time() - ingestion_start, 2)

    # COMBINE TEXT + OCR TEXT (multimodal support)
    combined_text = document_data.get("text", "") + "\n\n" + document_data.get("image_text", "")

    # CHUNKING
    chunking_start = time.time()
    chunks = chunk_text(combined_text)
    chunking_time = round(time.time() - chunking_start, 2)

    # CHUNK SUMMARIZATION
    chunk_start = time.time()
    chunk_summaries = summarize_chunks(
        chunks,
        mode=document_mode
    )
    chunk_time = round(time.time() - chunk_start, 2)

    # SEMANTIC SECTION BUILDING
    section_build_start = time.time()
    semantic_sections = build_semantic_sections(chunk_summaries)
    section_build_time = round(time.time() - section_build_start, 2)

    # SECTION SUMMARIZATION
    section_start = time.time()
    section_summaries = []

    for section in semantic_sections:
        section_summary = summarize_section(
            section["section_chunks"],
            section["section_id"]
        )

        section_summary["covered_chunk_ids"] = section["covered_chunk_ids"]
        section_summaries.append(section_summary)

    section_time = round(time.time() - section_start, 2)

    # EXECUTIVE SUMMARY
    executive_start = time.time()

    try:
        executive_summary = generate_executive_summary(
            section_summaries,
            mode=document_mode
        )

        executive_summary.setdefault(
            "tldr",
            "Executive TLDR generation failed."
        )

    except Exception as e:
        print("⚠️ Executive generation failed:", str(e))
        executive_summary = {
            "executive_summary": "Executive summary failed.",
            "key_points": [],
            "risks_action_items": [],
            "tldr": "Executive TLDR unavailable."
        }

    

    executive_time = round(time.time() - executive_start, 2)

    # FINAL ASSEMBLY
    final_output = assemble_document(
        executive_output=executive_summary,
        section_outputs=section_summaries,
        chunk_outputs=chunk_summaries,
        total_chunks=len(chunks)
    )

    # MEANING COVERAGE
    meaning_score = compute_meaning_coverage(
        section_summaries,
        executive_summary.get("executive_summary", "")
    )

    total_time = round(time.time() - total_start, 2)

    # RESPONSE
    response = final_output.model_dump()
    # ENTITY EXTRACTION
    entity_text = executive_summary.get("executive_summary", "")

    entities = extract_entities(entity_text)

    response["entities"] = entities

    response["performance"] = {
        "ingestion_time_sec": ingestion_time,
        "chunking_time_sec": chunking_time,
        "chunk_summarization_time_sec": chunk_time,
        "section_build_time_sec": section_build_time,
        "section_summarization_time_sec": section_time,
        "executive_time_sec": executive_time,
        "total_time_sec": total_time
    }

    response["document_summary"]["meaning_coverage_score"] = meaning_score
    response["document_summary"]["mode_used"] = document_mode

    # Clean up temp file
    os.remove(temp_path)

    return response