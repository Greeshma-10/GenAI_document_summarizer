"""
Graph Query Router — FastAPI endpoints

Endpoints:
  GET  /graph/query              — neighbours of an entity
  GET  /graph/path               — shortest path between two entities
  GET  /graph/type/{entity_type} — all entities of a given type
  GET  /graph/subgraph           — N-hop neighbourhood
  POST /graph/ask                — natural language query
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from .graph_service import GraphService
from v2.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graph", tags=["Graph Queries"])

_service: Optional[GraphService] = None


def get_service() -> GraphService:
    global _service
    if _service is None:
        _service = GraphService()
    return _service


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class NLQueryRequest(BaseModel):
    question: str
    limit: int = 20

class NLQueryResponse(BaseModel):
    question: str
    cypher: Optional[str]
    results: list
    count: int
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /graph/query?entity=Transformer&limit=10
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/query")
def query_entity(
    entity: str = Query(..., description="Entity name to look up"),
    limit:  int = Query(10,  description="Max results", ge=1, le=100)
):
    """Get direct relationships of an entity."""
    logger.info("Graph query | entity=%s | limit=%d", entity, limit)
    svc     = get_service()
    results = svc.query_entity(entity, limit)
    if not results:
        raise HTTPException(404, f"No relationships found for '{entity}'")
    return {"entity": entity, "count": len(results), "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /graph/path?from=BLEU&to=Transformer
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/path")
def query_path(
    from_entity: str = Query(..., alias="from", description="Start entity"),
    to_entity:   str = Query(..., alias="to",   description="End entity"),
):
    """Find the shortest connection path between two entities."""
    logger.info("Graph path | from=%s | to=%s", from_entity, to_entity)
    svc    = get_service()
    result = svc.query_path(from_entity, to_entity)
    if not result["found"]:
        raise HTTPException(
            404,
            f"No path found between '{from_entity}' and '{to_entity}' within 6 hops"
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /graph/type/MODEL?limit=20
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/type/{entity_type}")
def query_by_type(
    entity_type: str,
    limit: int = Query(30, description="Max results", ge=1, le=100)
):
    """Get all entities of a given type and their relationships."""
    valid_types = {"MODEL", "DATASET", "METRIC", "ORGANIZATION", "TASK", "CONCEPT"}
    if entity_type.upper() not in valid_types:
        raise HTTPException(
            400,
            f"Invalid type '{entity_type}'. Must be one of: {', '.join(sorted(valid_types))}"
        )
    logger.info("Graph by_type | type=%s | limit=%d", entity_type, limit)
    svc     = get_service()
    results = svc.query_by_type(entity_type, limit)
    if not results:
        raise HTTPException(404, f"No entities of type '{entity_type}' found")
    return {"type": entity_type.upper(), "count": len(results), "entities": results}


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /graph/subgraph?entity=Transformer&depth=2
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/subgraph")
def query_subgraph(
    entity: str = Query(..., description="Root entity name"),
    depth:  int = Query(2,   description="Hop depth (1-4)", ge=1, le=4),
    limit:  int = Query(50,  description="Max edges returned", ge=1, le=200)
):
    """Return all nodes and edges within N hops of an entity."""
    logger.info("Graph subgraph | entity=%s | depth=%d", entity, depth)
    svc    = get_service()
    result = svc.query_subgraph(entity, depth, limit)
    if result["node_count"] == 0:
        raise HTTPException(404, f"Entity '{entity}' not found in graph")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. POST /graph/ask
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/ask", response_model=NLQueryResponse)
def query_natural_language(request: NLQueryRequest):
    """Natural language graph query — converts question to Cypher and runs it."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    logger.info("NL graph query: %s", request.question)
    svc    = get_service()
    result = svc.query_natural_language(request.question, request.limit)

    if result.get("error") and not result["results"]:
        logger.warning("NL query failed: %s", result["error"])
        raise HTTPException(500, result["error"])

    return NLQueryResponse(**result)