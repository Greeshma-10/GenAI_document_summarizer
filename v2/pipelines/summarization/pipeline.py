from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary

from v2.pipelines.entity_extraction.entity_extractor import extract_entities

from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.graph.entity_utils import flatten_entities

from v2.logging_config import get_logger


logger = get_logger(__name__)


def run_summarization_pipeline(text: str, mode: str = "research"):

    # -----------------------------------------
    # STEP 1: Chunk document
    # -----------------------------------------
    chunks = chunk_text(text)

    # -----------------------------------------
    # STEP 2: Summarize chunks
    # -----------------------------------------
    chunk_summaries = summarize_chunks(chunks, mode=mode)

    # -----------------------------------------
    # STEP 3: Build semantic sections
    # -----------------------------------------
    sections = build_semantic_sections(chunk_summaries, mode=mode)

    # -----------------------------------------
    # STEP 4: Summarize sections
    # -----------------------------------------
    section_summaries = []

    for section in sections:

        section_summary = summarize_section(
            section["section_chunks"],
            section["section_id"]
        )

        section_summaries.append(section_summary)

    # -----------------------------------------
    # STEP 5: Executive summary
    # -----------------------------------------
    executive = generate_executive_summary(section_summaries, mode=mode)

    # -----------------------------------------
    # STEP 6: Knowledge Graph Generation
    # -----------------------------------------

    relation_extractor = RelationExtractor()
    graph_builder = GraphBuilder()
    graph_builder.clear_graph()

    all_triples = []

    for section in sections:

        section_text = " ".join(
            chunk["summary"] if isinstance(chunk, dict) else chunk
            for chunk in section["section_chunks"]
        )

        entities_dict = extract_entities(section_text, mode)
        entity_list = flatten_entities(entities_dict)

        logger.debug(f"Entities: {entity_list}")

        if not entity_list:
            continue

        triples = relation_extractor.extract_relations(
            section_text,
            entity_list
        )

        logger.debug(f"Triples: {triples}")

        if triples:
            all_triples.extend(triples)

    # -----------------------------------------
    # Insert triples into Neo4j
    # -----------------------------------------

    if all_triples:

        unique_triples = list(
            {
                (t.subject, t.relation, t.object): t
                for t in all_triples
            }.values()
        )

        logger.info(f"Triples generated: {len(unique_triples)}")

        graph_builder.insert_triples(unique_triples)

    else:
        logger.warning("No triples generated from document")

    # -----------------------------------------
    # Return pipeline output
    # -----------------------------------------

    return {
        "sections": section_summaries,
        "executive_summary": executive,
        "triples_created": len(all_triples)
    }