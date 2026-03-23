"""
FastAPI Application Entry Point

Endpoints:
  POST /summarize          — full document summarisation pipeline
  POST /entities           — entity extraction only
  POST /graph/build        — knowledge graph construction
  GET  /graph/query        — structured graph query
  POST /graph/ask          — natural language graph query
  POST /fact/verify        — single claim verification
  POST /fact/verify/batch  — batch claim verification
  POST /evaluate           — full pipeline evaluation
  GET  /search             — semantic vector search
"""

import os
import re
import tempfile
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

# Ingestion
from v2.ingestion.document_parser import parse_document

# Summarisation pipeline
from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.document_assembler import assemble_document

# Evaluation
from v2.pipelines.evaluation.meaning_evaluator import compute_coverage_score, run_full_evaluation

# Entity extraction
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.entity_pipeline import run_entity_pipeline

# Knowledge graph
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.graph.graph_service import GraphService
from v2.graph.fact_verifier import FactVerifier
from v2.graph.vector_store import VectorStore
from v2.graph.schema import CATEGORY_TO_SCHEMA_TYPE
from v2.graph.entity_utils import flatten_entities
from v2.pipelines.graph_pipeline import run_graph_pipeline

from v2.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Summarization & Knowledge Extraction API",
    version="2.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Service initialisation
# ─────────────────────────────────────────────────────────────────────────────

graph_service = GraphService()

try:
    vector_store  = VectorStore()
    fact_verifier = FactVerifier(graph_service, vector_store=vector_store)
    logger.info("Pinecone vector store initialised successfully")
except Exception:
    logger.warning("Pinecone unavailable — falling back to keyword-only fact verification")
    vector_store  = None
    fact_verifier = FactVerifier(graph_service)


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class NLQueryRequest(BaseModel):
    question: str
    limit:    int = 20

class FactCheckRequest(BaseModel):
    claim:       str
    source_text: Optional[str] = ""

class BatchFactCheckRequest(BaseModel):
    claims:      List[str]
    source_text: Optional[str] = ""

class EvaluationRequest(BaseModel):
    section_summaries:  List[dict]
    executive_summary:  str
    extracted_entities: Dict[str, List[str]]
    source_text:        Optional[str] = ""
    reference_entities: Optional[Dict[str, List[str]]] = None
    max_claims:         int = 15


# ─────────────────────────────────────────────────────────────────────────────
# Dedup helper (shared by /summarize and /graph/build)
# ─────────────────────────────────────────────────────────────────────────────

_INVERSE_PAIRS         = {"USES": "USED_IN", "USED_IN": "USES"}
_SUBJECT_MUST_BE_TYPED = {
    "DEVELOPED_BY", "PROPOSED_BY", "TRAINED_ON",
    "EVALUATED_ON", "APPLIED_TO", "PART_OF",
}

def _dedup_triples(all_triples, type_map):
    seen_keys      = set()
    unique_triples = []

    for t in all_triples:
        subj = t.subject.strip()
        obj  = t.object.strip()
        rel  = t.relation

        key = (subj.lower(), rel, obj.lower())
        if key in seen_keys:
            continue

        inverse_rel = _INVERSE_PAIRS.get(rel)
        if inverse_rel and (obj.lower(), inverse_rel, subj.lower()) in seen_keys:
            seen_keys.add(key)
            continue

        if rel in _SUBJECT_MUST_BE_TYPED and subj.lower() not in type_map:
            continue

        if subj.lower() == obj.lower():
            continue

        if rel in {"DEVELOPED_BY", "PROPOSED_BY"}:
            if type_map.get(obj.lower().strip(), "Unknown") not in {"ORGANIZATION", "Unknown"}:
                continue

        if re.search(r'\[|\bJournal\b|\bInternational\b|\bConference\b', subj, re.IGNORECASE):
            continue

        seen_keys.add(key)
        unique_triples.append(t)

    return unique_triples


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_mode(mode: str) -> str:
    return mode if mode in {"academic", "research"} else "academic"


async def _save_upload(file: UploadFile) -> str:
    """Write an uploaded file to a temp path and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        return tmp.name


def _index_chunks_in_pinecone(chunks: List[str], filename: str) -> None:
    """Best-effort Pinecone indexing — logs on failure, never raises."""
    if not vector_store:
        return
    try:
        doc_id = VectorStore.make_doc_id(filename)
        vector_store.delete_document(doc_id)
        vector_store.index_chunks(chunks, doc_id)
        logger.info("Indexed %d chunks in Pinecone | doc_id=%s", len(chunks), doc_id)
    except Exception:
        logger.exception("Pinecone indexing failed | filename=%s", filename)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"message": "API is healthy and running!"}


# ─────────────────────────────────────────────────────────────────────────────
# Summarise
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    mode: str        = Form("academic"),
):
    mode        = _validate_mode(mode)
    total_start = time.time()
    temp_path   = await _save_upload(file)

    try:
        # Ingestion
        t0            = time.time()
        document_data = parse_document(temp_path)
        ingestion_time = round(time.time() - t0, 2)
        logger.info("Ingestion complete | time=%.2fs", ingestion_time)

        combined_text = (
            document_data.get("text", "") + "\n\n" +
            document_data.get("image_text", "")
        )

        # Chunking
        t0            = time.time()
        chunks        = chunk_text(combined_text)
        chunking_time = round(time.time() - t0, 2)
        logger.info("Chunking complete | chunks=%d time=%.2fs", len(chunks), chunking_time)

        _index_chunks_in_pinecone(chunks, file.filename)

        # Chunk summarisation
        t0              = time.time()
        chunk_summaries = summarize_chunks(chunks, mode=mode)
        chunk_time      = round(time.time() - t0, 2)
        logger.info("Chunk summarisation complete | time=%.2fs", chunk_time)

        # Semantic section building
        t0                 = time.time()
        semantic_sections  = build_semantic_sections(chunk_summaries)
        section_build_time = round(time.time() - t0, 2)
        logger.info("Section building complete | sections=%d time=%.2fs",
                    len(semantic_sections), section_build_time)

        # Section summarisation
        t0                = time.time()
        section_summaries = []
        for section in semantic_sections:
            section_summary = summarize_section(
                section["section_chunks"],
                section["section_id"],
            )
            section_summary["covered_chunk_ids"] = section["covered_chunk_ids"]
            section_summaries.append(section_summary)
        section_time = round(time.time() - t0, 2)
        logger.info("Section summarisation complete | time=%.2fs", section_time)

        # Executive summary
        t0 = time.time()
        try:
            executive_summary = generate_executive_summary(section_summaries, mode=mode)
            executive_summary.setdefault("tldr", "Executive TLDR generation failed.")
        except Exception:
            logger.exception("Executive summary generation failed")
            executive_summary = {
                "executive_summary": "Executive summary failed.",
                "key_points":        [],
                "risks_action_items": [],
                "tldr":              "Executive TLDR unavailable.",
            }
        executive_time = round(time.time() - t0, 2)
        logger.info("Executive summary complete | time=%.2fs", executive_time)

        final_output  = assemble_document(
            executive_output=executive_summary,
            section_outputs=section_summaries,
            chunk_outputs=chunk_summaries,
            total_chunks=len(chunks),
        )
        meaning_score = compute_coverage_score(
            section_summaries,
            executive_summary.get("executive_summary", ""),
        )

        # Entity extraction — from section summaries
        merged_entities: dict = {}

        
        for sec in section_summaries:
            sec_text = sec.get("section_summary", "")
            if not sec_text:
                continue
            extracted = extract_entities(sec_text)
            for category, values in extracted.items():
                merged_entities.setdefault(category, []).extend(values)

        for category in merged_entities:
            merged_entities[category] = list(dict.fromkeys(merged_entities[category]))

        entity_dict = flatten_entities(merged_entities)

        # Knowledge graph — from section summaries
        relation_extractor = RelationExtractor()
        graph_builder      = GraphBuilder()
        all_triples        = []

        type_map = {
            v.lower().strip(): CATEGORY_TO_SCHEMA_TYPE.get(category, "CONCEPT")
            for category, values in entity_dict.items()
            for v in values
        }

        for sec in section_summaries:
            sec_text = sec.get("section_summary", "")
            if sec_text:
                all_triples.extend(
                    relation_extractor.extract_relations(sec_text, entity_dict)
                )

        unique_triples = _dedup_triples(all_triples, type_map)

        try:
            graph_builder.clear_graph()
            graph_builder.insert_triples(unique_triples)
            logger.info("Inserted %d triples into Neo4j", len(unique_triples))
        except Exception:
            logger.exception("Neo4j triple insertion failed")

        total_time = round(time.time() - total_start, 2)
        logger.info("Summarisation pipeline complete | total_time=%.2fs", total_time)

        response = final_output.model_dump()
        response["entities"] = merged_entities
        response["graph"]    = {"triples_inserted": len(unique_triples)}
        response["performance"] = {
            "ingestion_time_sec":             ingestion_time,
            "chunking_time_sec":              chunking_time,
            "chunk_summarization_time_sec":   chunk_time,
            "section_build_time_sec":         section_build_time,
            "section_summarization_time_sec": section_time,
            "executive_time_sec":             executive_time,
            "total_time_sec":                 total_time,
        }
        response["document_summary"]["meaning_coverage_score"] = meaning_score
        response["document_summary"]["mode_used"]              = mode

        return response

    finally:
        os.remove(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/entities")
async def extract_entities_api(
    file: UploadFile = File(...),
    mode: str        = Form("academic"),
):
    mode      = _validate_mode(mode)
    temp_path = await _save_upload(file)
    try:
        return run_entity_pipeline(temp_path, mode=mode)
    finally:
        os.remove(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Graph build
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/graph/build")
async def build_graph_api(
    file: UploadFile = File(...),
    mode: str        = Form("academic"),
):
    mode      = _validate_mode(mode)
    temp_path = await _save_upload(file)
    try:
        if vector_store:
            try:
                from v2.ingestion.document_parser import build_document_text
                parsed   = parse_document(temp_path)
                raw_text = build_document_text(parsed)
                chunks   = chunk_text(raw_text)
                _index_chunks_in_pinecone(chunks, file.filename)
            except Exception:
                logger.exception("Pinecone pre-indexing failed during graph build")

        return run_graph_pipeline(temp_path, mode=mode)
    finally:
        os.remove(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Graph query
# ─────────────────────────────────────────────────────────────────────────────

_VALID_ENTITY_TYPES = {"MODEL", "DATASET", "METRIC", "ORGANIZATION", "TASK", "CONCEPT"}

@app.get("/graph/query")
def query_graph(
    type:        str           = Query("neighbours"),
    entity:      Optional[str] = Query(None),
    from_entity: Optional[str] = Query(None, alias="from"),
    to_entity:   Optional[str] = Query(None, alias="to"),
    entity_type: Optional[str] = Query(None),
    depth:       int           = Query(2),
    limit:       int           = Query(20),
):
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
            if entity_type.upper() not in _VALID_ENTITY_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail="'entity_type' must be one of %s" % _VALID_ENTITY_TYPES,
                )
            return graph_service.query("by_type", entity_type=entity_type, limit=limit)

        elif type == "subgraph":
            if not entity:
                raise HTTPException(status_code=400, detail="'entity' param required")
            return graph_service.query("subgraph", entity=entity, depth=depth, limit=limit)

        else:
            raise HTTPException(
                status_code=400,
                detail="'type' must be one of: neighbours | path | by_type | subgraph",
            )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Graph query failed | type=%s entity=%s", type, entity)
        raise HTTPException(status_code=500, detail="Graph query failed")


# ─────────────────────────────────────────────────────────────────────────────
# Natural language graph query
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/graph/ask")
def query_natural_language(request: NLQueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = graph_service.query_nl(request.question, request.limit)
        if result.get("error") and not result["results"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Natural language graph query failed | question=%s",
                         request.question[:80])
        raise HTTPException(status_code=500, detail="Natural language query failed")


# ─────────────────────────────────────────────────────────────────────────────
# Fact verification — single
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/fact/verify")
def verify_fact(request: FactCheckRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    try:
        return fact_verifier.verify(request.claim, request.source_text or "")
    except Exception:
        logger.exception("Fact verification failed | claim=%s", request.claim[:80])
        raise HTTPException(status_code=500, detail="Fact verification failed")


# ─────────────────────────────────────────────────────────────────────────────
# Fact verification — batch
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/fact/verify/batch")
def verify_facts_batch(request: BatchFactCheckRequest):
    if not request.claims:
        raise HTTPException(status_code=400, detail="Claims list cannot be empty")
    try:
        results      = fact_verifier.verify_batch(request.claims, request.source_text or "")
        supported    = sum(1 for r in results if r["verdict"] == "SUPPORTED")
        contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
        unverified   = sum(1 for r in results if r["verdict"] == "UNVERIFIED")
        return {
            "total":        len(results),
            "supported":    supported,
            "contradicted": contradicted,
            "unverified":   unverified,
            "results":      results,
        }
    except Exception:
        logger.exception("Batch fact verification failed | claim_count=%d", len(request.claims))
        raise HTTPException(status_code=500, detail="Batch fact verification failed")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    try:
        return run_full_evaluation(
            section_summaries=request.section_summaries,
            executive_summary_text=request.executive_summary,
            extracted_entities=request.extracted_entities,
            fact_verifier=fact_verifier,
            source_text=request.source_text or "",
            reference_entities=request.reference_entities,
            max_claims=request.max_claims,
        )
    except Exception:
        logger.exception("Evaluation pipeline failed")
        raise HTTPException(status_code=500, detail="Evaluation failed")


# ─────────────────────────────────────────────────────────────────────────────
# Semantic search
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/search")
def semantic_search(
    query: str = Query(..., description="Natural language search query"),
    top_k: int = Query(5,   description="Number of results to return"),
):
    if not vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable — check PINECONE_API_KEY",
        )
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        results = vector_store.search(query, top_k=top_k)
        return {"query": query, "count": len(results), "results": results}
    except Exception:
        logger.exception("Semantic search failed | query=%s", query[:80])
        raise HTTPException(status_code=500, detail="Semantic search failed")