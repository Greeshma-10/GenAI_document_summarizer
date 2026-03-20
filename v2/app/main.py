from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import time
import tempfile
import os

# Ingestion
from v2.ingestion.document_parser import parse_document

# Summarization pipeline
from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.document_assembler import assemble_document

# Evaluation
from v2.pipelines.evaluation.meaning_evaluator import compute_meaning_coverage

# Entity extraction
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.entity_pipeline import run_entity_pipeline

# Knowledge graph
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.pipelines.graph_pipeline import run_graph_pipeline
from v2.graph.graph_service import GraphService

# Utilities — flatten_entities now returns the dict as-is (not a flat list)
from v2.graph.entity_utils import flatten_entities


app = FastAPI(
    title="Document Summarization & Knowledge Extraction API",
    version="2.0"
)

graph_service = GraphService()


class NLQueryRequest(BaseModel):
    question: str
    limit: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"message": "API is healthy and running!"}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    mode: str = Form("academic")
):
    if mode not in ["academic", "research"]:
        mode = "academic"

    document_mode = mode
    total_start = time.time()

    ingestion_start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    document_data = parse_document(temp_path)
    ingestion_time = round(time.time() - ingestion_start, 2)

    combined_text = (
        document_data.get("text", "") + "\n\n" +
        document_data.get("image_text", "")
    )

    chunking_start = time.time()
    chunks = chunk_text(combined_text)
    chunking_time = round(time.time() - chunking_start, 2)

    chunk_start = time.time()
    chunk_summaries = summarize_chunks(chunks, mode=document_mode)
    chunk_time = round(time.time() - chunk_start, 2)

    section_build_start = time.time()
    semantic_sections = build_semantic_sections(chunk_summaries)
    section_build_time = round(time.time() - section_build_start, 2)

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

    executive_start = time.time()
    try:
        executive_summary = generate_executive_summary(
            section_summaries,
            mode=document_mode
        )
        executive_summary.setdefault("tldr", "Executive TLDR generation failed.")
    except Exception as e:
        print("⚠️ Executive generation failed:", str(e))
        executive_summary = {
            "executive_summary": "Executive summary failed.",
            "key_points": [],
            "risks_action_items": [],
            "tldr": "Executive TLDR unavailable."
        }
    executive_time = round(time.time() - executive_start, 2)

    final_output = assemble_document(
        executive_output=executive_summary,
        section_outputs=section_summaries,
        chunk_outputs=chunk_summaries,
        total_chunks=len(chunks)
    )

    meaning_score = compute_meaning_coverage(
        section_summaries,
        executive_summary.get("executive_summary", "")
    )

    # ── ENTITY EXTRACTION ────────────────────────────────────────────────────
    # Run extraction over ALL section summaries (not just the last one)
    # and merge results into a single categorized dict
    merged_entities: dict = {}
    for sec in section_summaries:
        sec_text = sec.get("section_summary", "")
        if not sec_text:
            continue
        extracted = extract_entities(sec_text, mode=document_mode)
        for category, values in extracted.items():
            if category not in merged_entities:
                merged_entities[category] = []
            merged_entities[category].extend(values)

    # Deduplicate within each category
    for category in merged_entities:
        merged_entities[category] = list(dict.fromkeys(merged_entities[category]))

    # flatten_entities cleans/filters — returns same dict structure
    # relation_extractor needs { "models": [...], "organizations": [...], ... }
    entity_dict = flatten_entities(merged_entities)

    # ── KNOWLEDGE GRAPH ───────────────────────────────────────────────────────
    relation_extractor = RelationExtractor()
    graph_builder = GraphBuilder()
    all_triples = []

    for sec in section_summaries:
        sec_text = sec.get("section_summary", "")
        if not sec_text:
            continue
        triples = relation_extractor.extract_relations(sec_text, entity_dict)
        all_triples.extend(triples)

    try:
        graph_builder.clear_graph()
        graph_builder.insert_triples(all_triples)
        print(f"📥 Inserted {len(all_triples)} triples into Neo4j")
    except Exception as e:
        print("⚠️ Neo4j insertion failed:", str(e))

    total_time = round(time.time() - total_start, 2)

    response = final_output.model_dump()
    response["entities"] = merged_entities       # return full categorized dict
    response["graph"] = {"triples_inserted": len(all_triples)}
    response["performance"] = {
        "ingestion_time_sec":            ingestion_time,
        "chunking_time_sec":             chunking_time,
        "chunk_summarization_time_sec":  chunk_time,
        "section_build_time_sec":        section_build_time,
        "section_summarization_time_sec": section_time,
        "executive_time_sec":            executive_time,
        "total_time_sec":                total_time,
    }
    response["document_summary"]["meaning_coverage_score"] = meaning_score
    response["document_summary"]["mode_used"] = document_mode

    os.remove(temp_path)
    return response


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/entities")
async def extract_entities_api(
    file: UploadFile = File(...),
    mode: str = Form("academic")
):
    if mode not in ["academic", "research"]:
        mode = "academic"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        result = run_entity_pipeline(temp_path, mode=mode)
    finally:
        os.remove(temp_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILD
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/graph/build")
async def build_graph_api(
    file: UploadFile = File(...),
    mode: str = Form("academic")
):
    if mode not in ["academic", "research"]:
        mode = "academic"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        result = run_graph_pipeline(temp_path, mode=mode)
    finally:
        os.remove(temp_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH QUERY — unified: neighbours | path | by_type | subgraph
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/graph/query")
def query_graph(
    type:        str           = Query("neighbours", description="neighbours | path | by_type | subgraph"),
    entity:      Optional[str] = Query(None, description="Entity name (neighbours / subgraph)"),
    from_entity: Optional[str] = Query(None, alias="from", description="Start entity (path)"),
    to_entity:   Optional[str] = Query(None, alias="to",   description="End entity (path)"),
    entity_type: Optional[str] = Query(None, description="MODEL | DATASET | METRIC | ORGANIZATION | TASK | CONCEPT"),
    depth:       int           = Query(2,    description="Hop depth for subgraph (1–4)"),
    limit:       int           = Query(20,   description="Max results"),
):
    """
    Unified graph query endpoint.

    Examples:
      /graph/query?type=neighbours&entity=Transformer
      /graph/query?type=path&from=BLEU&to=Google+Brain
      /graph/query?type=by_type&entity_type=MODEL
      /graph/query?type=subgraph&entity=Transformer&depth=2
    """
    try:
        if type == "neighbours":
            if not entity:
                raise HTTPException(status_code=400, detail="'entity' param required")
            return graph_service.query("neighbours", entity=entity, limit=limit)

        elif type == "path":
            if not from_entity or not to_entity:
                raise HTTPException(status_code=400, detail="'from' and 'to' params required")
            return graph_service.query("path", from_entity=from_entity, to_entity=to_entity)

        elif type == "by_type":
            if not entity_type:
                raise HTTPException(status_code=400, detail="'entity_type' param required")
            valid = {"MODEL", "DATASET", "METRIC", "ORGANIZATION", "TASK", "CONCEPT"}
            if entity_type.upper() not in valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"entity_type must be one of {valid}"
                )
            return graph_service.query("by_type", entity_type=entity_type, limit=limit)

        elif type == "subgraph":
            if not entity:
                raise HTTPException(status_code=400, detail="'entity' param required")
            return graph_service.query("subgraph", entity=entity, depth=depth, limit=limit)

        else:
            raise HTTPException(
                status_code=400,
                detail="'type' must be: neighbours | path | by_type | subgraph"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE GRAPH QUERY
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/graph/ask")
def query_natural_language(request: NLQueryRequest):
    """
    Ask anything in plain English — converted to Cypher and run on the graph.

    Examples:
      { "question": "Which models were developed by Google Brain?" }
      { "question": "How is BLEU related to the Transformer?" }
      { "question": "What concepts does the Transformer use?" }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = graph_service.query_nl(request.question, request.limit)
        if result.get("error") and not result["results"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))