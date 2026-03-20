from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
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
from v2.pipelines.evaluation.meaning_evaluator import compute_coverage_score

# Entity extraction
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.entity_pipeline import run_entity_pipeline

# Knowledge graph
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.pipelines.graph_pipeline import run_graph_pipeline
from v2.graph.graph_service import GraphService
from v2.graph.fact_verifier import FactVerifier

# Utilities
from v2.graph.entity_utils import flatten_entities

from v2.pipelines.evaluation.meaning_evaluator import run_full_evaluation
from typing import Optional, List, Dict


app = FastAPI(
    title="Document Summarization & Knowledge Extraction API",
    version="2.0"
)

graph_service = GraphService()
fact_verifier = FactVerifier(graph_service)


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────
class NLQueryRequest(BaseModel):
    question: str
    limit: int = 20

class FactCheckRequest(BaseModel):
    claim: str
    source_text: Optional[str] = ""

class BatchFactCheckRequest(BaseModel):
    claims: List[str]
    source_text: Optional[str] = ""


class EvaluationRequest(BaseModel):
    # Required
    section_summaries:      List[dict]
    executive_summary:      str
    extracted_entities:     Dict[str, List[str]]
    # Optional
    source_text:            Optional[str] = ""
    reference_entities:     Optional[Dict[str, List[str]]] = None
    max_claims:             int = 15
 

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

    meaning_score = compute_coverage_score(
        section_summaries,
        executive_summary.get("executive_summary", "")
    )

    # ── ENTITY EXTRACTION ────────────────────────────────────────────────────
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

    for category in merged_entities:
        merged_entities[category] = list(dict.fromkeys(merged_entities[category]))

    entity_dict = flatten_entities(merged_entities)

    # ── KNOWLEDGE GRAPH ───────────────────────────────────────────────────────
    relation_extractor = RelationExtractor()
    graph_builder = GraphBuilder()
    all_triples = []

    _INVERSE_PAIRS = {"USES": "USED_IN", "USED_IN": "USES"}

    for sec in section_summaries:
        sec_text = sec.get("section_summary", "")
        if not sec_text:
            continue
        triples = relation_extractor.extract_relations(sec_text, entity_dict)
        all_triples.extend(triples)

    # Deduplicate — exact + semantic inverse
    seen_keys = set()
    unique_triples = []
    for t in all_triples:
        key = (t.subject.lower(), t.relation, t.object.lower())
        if key in seen_keys:
            continue
        inverse_rel = _INVERSE_PAIRS.get(t.relation)
        if inverse_rel:
            inverse_key = (t.object.lower(), inverse_rel, t.subject.lower())
            if inverse_key in seen_keys:
                seen_keys.add(key)
                continue
        seen_keys.add(key)
        unique_triples.append(t)

    try:
        graph_builder.clear_graph()
        graph_builder.insert_triples(unique_triples)
        print(f"📥 Inserted {len(unique_triples)} triples into Neo4j")
    except Exception as e:
        print("⚠️ Neo4j insertion failed:", str(e))
    total_time = round(time.time() - total_start, 2)

    response = final_output.model_dump()
    response["entities"] = merged_entities
    response["graph"] = {"triples_inserted": len(all_triples)}
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


# ─────────────────────────────────────────────────────────────────────────────
# FACT VERIFICATION — single claim
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/fact/verify")
def verify_fact(request: FactCheckRequest):
    """
    Verify a single claim against the knowledge graph and source text.

    Returns verdict: SUPPORTED | CONTRADICTED | UNVERIFIED

    Example:
      {
        "claim": "The Transformer was developed by Google Brain",
        "source_text": "..."
      }
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    try:
        result = fact_verifier.verify(request.claim, request.source_text or "")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# FACT VERIFICATION — batch
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/fact/verify/batch")
def verify_facts_batch(request: BatchFactCheckRequest):
    """
    Verify multiple claims at once.

    Example:
      {
        "claims": [
          "The Transformer was developed by Google Brain",
          "BLEU is used to evaluate translation",
          "The Transformer uses convolutional layers"
        ],
        "source_text": "..."
      }
    """
    if not request.claims:
        raise HTTPException(status_code=400, detail="Claims list cannot be empty")
    try:
        results = fact_verifier.verify_batch(
            request.claims, request.source_text or ""
        )
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # 3. Add this endpoint:
 
@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    """
    Run full evaluation on a processed document.
 
    Computes:
      - coverage_score     : semantic similarity, executive vs sections (0-100)
      - factual_accuracy   : % of claims supported by knowledge graph (0-100)
      - entity_accuracy    : entity extraction quality (precision/recall/F1
                             if reference provided, else self-consistency score)
 
    Minimal example (no reference entities):
      {
        "section_summaries": [
          {"section_summary": "The Transformer was developed by Google Brain..."}
        ],
        "executive_summary": "The Transformer architecture introduced by Google Brain...",
        "extracted_entities": {
          "models": ["Transformer"],
          "organizations": ["Google Brain"],
          "metrics": ["BLEU"]
        }
      }
 
    With reference entities (for precision/recall):
      {
        ...,
        "reference_entities": {
          "models": ["Transformer", "BERT"],
          "organizations": ["Google Brain"]
        }
      }
    """
    try:
        result = run_full_evaluation(
            section_summaries=request.section_summaries,
            executive_summary_text=request.executive_summary,
            extracted_entities=request.extracted_entities,
            fact_verifier=fact_verifier,
            source_text=request.source_text or "",
            reference_entities=request.reference_entities,
            max_claims=request.max_claims,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 