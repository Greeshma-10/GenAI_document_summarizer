from v2.ingestion.document_parser import parse_document, build_document_text
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.summarization.chunking import chunk_text
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder


# Garbage values the LLM hallucinates as organization/developer names
_GARBAGE = {
    "none", "model", "models", "organization", "organizations",
    "language modeling", "the authors", "authors", "unknown"
}


def _is_garbage(value: str) -> bool:
    return not value or value.strip().lower() in _GARBAGE


def run_graph_pipeline(file_path, mode="academic"):

    # ── INGESTION ─────────────────────────────────────────────────────────────
    parsed_doc = parse_document(file_path)
    full_text  = build_document_text(parsed_doc)

    # ── CHUNKING ──────────────────────────────────────────────────────────────
    chunks = chunk_text(full_text)

    # ── ENTITY EXTRACTION ─────────────────────────────────────────────────────
    all_entities = {
        "models":        [],
        "datasets":      [],
        "metrics":       [],
        "organizations": [],
        "tasks":         [],
        "key_concepts":  []
    }

    # Pass 1: extract from every chunk
    for i, chunk in enumerate(chunks):
        chunk_entities = extract_entities(chunk, mode=mode)
        print(f"🧪 Chunk {i+1}/{len(chunks)}: "
              f"{ {k: v for k, v in chunk_entities.items() if v} }")
        for key in all_entities:
            all_entities[key].extend(chunk_entities.get(key, []))

    # Pass 2: dedicated organization extraction from first 2 chunks
    # (author affiliations are almost always in the paper header)
    org_extraction_text = "\n\n".join(chunks[:2])
    org_result = extract_entities(org_extraction_text, mode=mode)
    extra_orgs = org_result.get("organizations", [])
    if extra_orgs:
        print(f"🏢 Extra orgs from header pass: {extra_orgs}")
        all_entities["organizations"].extend(extra_orgs)

    # Deduplicate + remove garbage values from every category
    for key in all_entities:
        seen = []
        for v in dict.fromkeys(all_entities[key]):   # dedup preserving order
            if not _is_garbage(v):
                seen.append(v)
        all_entities[key] = seen

    print(f"\n📋 FINAL MERGED ENTITIES:")
    for key, values in all_entities.items():
        if values:
            print(f"  {key}: {values}")

    if not all_entities.get("organizations"):
        print("⚠️  No organizations found — DEVELOPED_BY triples will be skipped")

    # ── RELATION EXTRACTION ───────────────────────────────────────────────────
    relation_extractor = RelationExtractor()
    all_triples = []

    for i, chunk in enumerate(chunks):
        triples = relation_extractor.extract_relations(chunk, all_entities)
        # Filter out triples with garbage objects right here
        clean = [
            t for t in triples
            if not _is_garbage(t.object) and not _is_garbage(t.subject)
        ]
        print(f"🔗 Chunk {i+1} → {len(clean)} triples (filtered from {len(triples)})")
        all_triples.extend(clean)

    # Deduplicate triples
    seen_keys = set()
    unique_triples = []
    for t in all_triples:
        key = (t.subject, t.relation, t.object)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_triples.append(t)

    print(f"\n✅ Total unique triples: {len(unique_triples)}")

    # ── GRAPH INSERTION ───────────────────────────────────────────────────────
    graph_builder = GraphBuilder()

    try:
        graph_builder.clear_graph()
        graph_builder.insert_triples(unique_triples)
    except Exception as e:
        print("⚠️ Graph insertion failed:", str(e))
        return {"status": "failed", "error": str(e)}

    return {
        "status":        "success",
        "num_entities":  sum(len(v) for v in all_entities.values()),
        "num_relations": len(unique_triples),
        "entities":      all_entities,
    }