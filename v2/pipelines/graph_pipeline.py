from v2.ingestion.document_parser import parse_document, build_document_text
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.summarization.chunking import chunk_text
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder


MAX_CHUNKS = 10

_GARBAGE = {
    "none", "model", "models", "organization", "organizations",
    "language modeling", "the authors", "authors", "unknown"
}

# Relations that are semantic inverses of each other
# If (A)-[REL]->(B) exists, drop (B)-[INVERSE]->(A)
_INVERSE_PAIRS = {
    "USES":    "USED_IN",
    "USED_IN": "USES",
}

# Relations where the subject must be a known typed entity (MODEL/ORG/DATASET/METRIC)
# Prevents noise like "attention key and value dimensions -[PROPOSED_BY]-> ..."
_SUBJECT_MUST_BE_TYPED = {
    "DEVELOPED_BY", "PROPOSED_BY", "TRAINED_ON",
    "EVALUATED_ON", "APPLIED_TO", "PART_OF"
}


def _is_garbage(value: str) -> bool:
    return not value or value.strip().lower() in _GARBAGE


def _is_typed_entity(name: str, entity_type_map: dict) -> bool:
    """Check if an entity name exists in the known typed entity map."""
    name_lower = name.lower().strip()
    for k in entity_type_map:
        if k.lower().strip() == name_lower:
            return True
        # partial match
        if name_lower in k.lower() or k.lower() in name_lower:
            return True
    return False


def _build_type_map(all_entities: dict) -> dict:
    """Flat map of entity_name → type for quick lookup."""
    from v2.graph.schema import CATEGORY_TO_SCHEMA_TYPE
    result = {}
    for category, values in all_entities.items():
        etype = CATEGORY_TO_SCHEMA_TYPE.get(category, "CONCEPT")
        for v in values:
            result[v.lower().strip()] = etype
    return result


def run_graph_pipeline(file_path, mode="academic"):

    # ── INGESTION ─────────────────────────────────────────────────────────────
    parsed_doc = parse_document(file_path)
    full_text  = build_document_text(parsed_doc)

    # ── CHUNKING ──────────────────────────────────────────────────────────────
    chunks = chunk_text(full_text)
    total  = len(chunks)
    chunks = chunks[:MAX_CHUNKS]
    print(f"📄 Processing {len(chunks)}/{total} chunks (MAX_CHUNKS={MAX_CHUNKS})")

    # ── ENTITY EXTRACTION ─────────────────────────────────────────────────────
    all_entities = {
        "models": [], "datasets": [], "metrics": [],
        "organizations": [], "tasks": [], "key_concepts": []
    }

    for i, chunk in enumerate(chunks):
        chunk_entities = extract_entities(chunk, mode=mode)
        print(f"🧪 Chunk {i+1}/{len(chunks)}: "
              f"{ {k: v for k, v in chunk_entities.items() if v} }")
        for key in all_entities:
            all_entities[key].extend(chunk_entities.get(key, []))

    # Pass 2: org extraction from header
    org_result = extract_entities("\n\n".join(chunks[:2]), mode=mode)
    extra_orgs = org_result.get("organizations", [])
    if extra_orgs:
        print(f"🏢 Extra orgs from header pass: {extra_orgs}")
        all_entities["organizations"].extend(extra_orgs)

    # Deduplicate + remove garbage
    for key in all_entities:
        seen = []
        for v in dict.fromkeys(all_entities[key]):
            if not _is_garbage(v):
                seen.append(v)
        all_entities[key] = seen

    print(f"\n📋 FINAL MERGED ENTITIES:")
    for key, values in all_entities.items():
        if values:
            print(f"  {key}: {values}")

    # Build flat type map for subject validation
    type_map = _build_type_map(all_entities)

    # ── RELATION EXTRACTION ───────────────────────────────────────────────────
    relation_extractor = RelationExtractor()
    all_triples = []

    for i, chunk in enumerate(chunks):
        triples = relation_extractor.extract_relations(chunk, all_entities)
        clean = [
            t for t in triples
            if not _is_garbage(t.object) and not _is_garbage(t.subject)
        ]
        print(f"🔗 Chunk {i+1} → {len(clean)} triples")
        all_triples.extend(clean)

    # ── DEDUPLICATION ─────────────────────────────────────────────────────────
    seen_keys  = set()
    unique_triples = []

    for t in all_triples:
        subj = t.subject.strip()
        obj  = t.object.strip()
        rel  = t.relation

        # 1. Skip exact duplicates
        key = (subj.lower(), rel, obj.lower())
        if key in seen_keys:
            continue

        # 2. Skip semantic inverse duplicates
        #    e.g. if (Transformer USES self-attention) already exists,
        #    drop (self-attention USED_IN Transformer)
        inverse_rel = _INVERSE_PAIRS.get(rel)
        if inverse_rel:
            inverse_key = (obj.lower(), inverse_rel, subj.lower())
            if inverse_key in seen_keys:
                print(f"   🔁 Skipping inverse duplicate: "
                      f"({subj}) -[{rel}]-> ({obj})")
                seen_keys.add(key)   # mark so we don't add the forward either
                continue

        # 3. For structural relations, subject must be a known typed entity
        #    Prevents noise like "attention key and value dimensions -[PROPOSED_BY]-> ..."
        if rel in _SUBJECT_MUST_BE_TYPED:
            if not _is_typed_entity(subj, type_map):
                print(f"   ⏭ Skipping: subject '{subj}' not a typed entity "
                      f"for relation {rel}")
                continue

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
        "status":           "success",
        "num_entities":     sum(len(v) for v in all_entities.values()),
        "num_relations":    len(unique_triples),
        "entities":         all_entities,
        "chunks_processed": len(chunks),
        "chunks_total":     total,
    }