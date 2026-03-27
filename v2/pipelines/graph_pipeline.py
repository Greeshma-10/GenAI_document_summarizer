from v2.ingestion.document_parser import parse_document, build_document_text
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.pipelines.summarization.chunking import chunk_text
from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.logging_config import get_logger


logger = get_logger(__name__)

_GARBAGE = {
    "none", "model", "models", "organization", "organizations",
    "language modeling", "the authors", "authors", "unknown"
}

_INVERSE_PAIRS = {
    "USES": "USED_IN",
    "USED_IN": "USES",
}

_SUBJECT_MUST_BE_TYPED = {
    "DEVELOPED_BY", "PROPOSED_BY", "TRAINED_ON",
    "EVALUATED_ON", "APPLIED_TO", "PART_OF"
}


def _is_garbage(value: str) -> bool:
    return not value or value.strip().lower() in _GARBAGE


def _is_typed_entity(name: str, entity_type_map: dict) -> bool:
    name_lower = name.lower().strip()
    for k in entity_type_map:
        if k.lower().strip() == name_lower:
            return True
        if name_lower in k.lower() or k.lower() in name_lower:
            return True
    return False


def _build_type_map(all_entities: dict) -> dict:
    from v2.graph.schema import CATEGORY_TO_SCHEMA_TYPE
    result = {}
    for category, values in all_entities.items():
        etype = CATEGORY_TO_SCHEMA_TYPE.get(category, "CONCEPT")
        for v in values:
            result[v.lower().strip()] = etype
    return result


def run_graph_pipeline(file_path, mode="academic"):

    parsed_doc = parse_document(file_path)
    full_text = build_document_text(parsed_doc)

    chunks = chunk_text(full_text)
    total = len(chunks)

    logger.info(f"Processing all {total} chunks (Bedrock — no cap)")

    all_entities = {
        "models": [], "datasets": [], "metrics": [],
        "organizations": [], "tasks": [], "key_concepts": []
    }

    for i, chunk in enumerate(chunks):
        chunk_entities = extract_entities(chunk)

        logger.debug(
            f"Chunk {i+1}/{total}: "
            f"{ {k: v for k, v in chunk_entities.items() if v} }"
        )

        for key in all_entities:
            all_entities[key].extend(chunk_entities.get(key, []))

    org_result = extract_entities("\n\n".join(chunks[:2]))
    extra_orgs = org_result.get("organizations", [])

    if extra_orgs:
        logger.info(f"Extra orgs from header pass: {extra_orgs}")
        all_entities["organizations"].extend(extra_orgs)

    for key in all_entities:
        seen = []
        for v in dict.fromkeys(all_entities[key]):
            if not _is_garbage(v):
                seen.append(v)
        all_entities[key] = seen

    logger.info("FINAL MERGED ENTITIES:")
    for key, values in all_entities.items():
        if values:
            logger.info(f"{key}: {values}")

    type_map = _build_type_map(all_entities)

    relation_extractor = RelationExtractor()
    all_triples = []

    for i, chunk in enumerate(chunks):
        triples = relation_extractor.extract_relations(chunk, all_entities)

        clean = [
            t for t in triples
            if not _is_garbage(t.object) and not _is_garbage(t.subject)
        ]

        logger.debug(f"Chunk {i+1}/{total} → {len(clean)} triples")

        all_triples.extend(clean)

    import re as _re
    seen_keys = set()
    unique_triples = []

    for t in all_triples:
        subj = t.subject.strip()
        obj = t.object.strip()
        rel = t.relation

        key = (subj.lower(), rel, obj.lower())
        if key in seen_keys:
            continue

        inverse_rel = _INVERSE_PAIRS.get(rel)
        if inverse_rel:
            inverse_key = (obj.lower(), inverse_rel, subj.lower())
            if inverse_key in seen_keys:
                logger.debug(
                    f"Skipping inverse duplicate: ({subj}) -[{rel}]-> ({obj})"
                )
                seen_keys.add(key)
                continue

        if rel in _SUBJECT_MUST_BE_TYPED:
            if not _is_typed_entity(subj, type_map):
                logger.debug(
                    f"Skipping: subject '{subj}' not a typed entity for {rel}"
                )
                continue

        if subj.lower() == obj.lower():
            continue

        if rel in {"DEVELOPED_BY", "PROPOSED_BY"}:
            obj_type = type_map.get(obj.lower(), "Unknown")
            if obj_type not in {"ORGANIZATION", "Unknown"}:
                continue

        if _re.search(r'\[|\bJournal\b|\bInternational\b|\bConference\b',
                      subj, _re.IGNORECASE):
            continue

        seen_keys.add(key)
        unique_triples.append(t)

    logger.info(f"Total unique triples: {len(unique_triples)}")

    graph_builder = GraphBuilder()

    try:
        graph_builder.clear_graph()
        graph_builder.insert_triples(unique_triples)
    except Exception as e:
        logger.error(f"Graph insertion failed: {str(e)}")
        return {"status": "failed", "error": str(e)}

    return {
        "status": "success",
        "num_entities": sum(len(v) for v in all_entities.values()),
        "num_relations": len(unique_triples),
        "entities": all_entities,
        "chunks_processed": total,
        "chunks_total": total,
    }