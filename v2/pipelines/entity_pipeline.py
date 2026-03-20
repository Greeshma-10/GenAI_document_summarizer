from v2.ingestion.document_parser import parse_document, build_document_text
from v2.pipelines.entity_extraction.entity_extractor import extract_entities
from v2.graph.entity_utils import flatten_entities
from v2.pipelines.summarization.chunking import chunk_text


# ----------------------------
# CLEANING FUNCTIONS
# ----------------------------

def clean_entities(entity_dict):
    cleaned = {}

    for key, values in entity_dict.items():
        normalized_list = []
        original_map = {}

        for v in values:
            norm = (
                v.strip()
                .lower()
                .replace("-", " ")
                .replace("_", " ")
            )
            norm = " ".join(norm.split())

            # Skip if exact duplicate
            if norm in normalized_list:
                continue

            # Controlled similarity check (only for long entities)
            is_duplicate = False
            for existing in normalized_list:
                if len(norm) > 10 and (norm in existing or existing in norm):
                    is_duplicate = True
                    break

            if not is_duplicate:
                normalized_list.append(norm)
                original_map[norm] = v.strip()

        # Preserve original formatting
        cleaned[key] = [original_map[n] for n in normalized_list]

    return cleaned

def filter_entities(entity_dict):
    filtered = {}

    for key, values in entity_dict.items():
        filtered[key] = [
            v for v in values
            if len(v.strip()) > 3 and "dataset" not in v.lower()
        ]

    return filtered


def normalize_entities(entity_dict):
    normalized = {}

    for key, values in entity_dict.items():
        normalized[key] = [
            v.strip().title() for v in values
        ]

    return normalized


# ----------------------------
# MAIN PIPELINE
# ----------------------------

def run_entity_pipeline(file_path, mode="academic"):

    # ----------------------------
    # INGESTION
    # ----------------------------
    parsed_doc = parse_document(file_path)

    # ----------------------------
    # BUILD FULL TEXT (TEXT + OCR)
    # ----------------------------
    full_text = build_document_text(parsed_doc)

    # 🔍 DEBUG
    print("\n🔍 TEXT LENGTH:", len(full_text))
    print("\n🔍 SAMPLE TEXT:\n", full_text[:500])

    # ----------------------------
    # CHUNKING
    # ----------------------------
    chunks = chunk_text(full_text)

    print("\n🔍 NUMBER OF CHUNKS:", len(chunks))

    # ----------------------------
    # ENTITY EXTRACTION (CHUNK-WISE)
    # ----------------------------
    all_entities = {
        "models": [],
        "datasets": [],
        "metrics": [],
        "organizations": [],
        "tasks": [],
        "key_concepts": []
    }

    for i, chunk in enumerate(chunks[:5]):  # limit chunks

        print(f"\n🧩 Processing chunk {i+1}...")

        chunk_entities = extract_entities(chunk, mode=mode)

        print("\n🧠 Chunk Entities:", chunk_entities)

        for key in all_entities:
            all_entities[key].extend(chunk_entities.get(key, []))

    # ----------------------------
    # CLEANING PIPELINE (IMPORTANT)
    # ----------------------------
    entities = clean_entities(all_entities)
    entities = filter_entities(entities)
    #entities = normalize_entities(entities)

    # ----------------------------
    # FLATTEN FOR FRONTEND
    # ----------------------------
    entity_list = flatten_entities(entities)

    print("\n✅ FINAL CLEANED ENTITIES:", entities)
    print("\n✅ FINAL ENTITY LIST:", entity_list)

    return {
        "entities": entities,
        "entity_list": entity_list
    }