# from v2.pipelines.summarization.chunking import chunk_text
# from v2.pipelines.summarization.summarizer import summarize_chunks
# from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
# from v2.pipelines.summarization.section_summarizer import summarize_section
# from v2.pipelines.summarization.executive_summarizer import generate_executive_summary
# from v2.pipelines.entity_extraction.entity_extractor import extract_entities
# from v2.graph.relation_extractor import RelationExtractor
# from v2.graph.graph_builder import GraphBuilder
# from v2.graph.entity_utils import flatten_entities


# def run_summarization_pipeline(text: str, mode: str = "research"):

#     # Step 1: chunk document
#     chunks = chunk_text(text)

#     # Step 2: summarize chunks
#     chunk_summaries = summarize_chunks(chunks, mode=mode)

#     # Step 3: semantic clustering
#     sections = build_semantic_sections(chunk_summaries, mode=mode)

#     # Step 4: summarize sections
#     section_summaries = []

#     for section in sections:

#         section_summary = summarize_section(
#             section["section_chunks"],
#             section["section_id"]
#         )

#         section_summaries.append(section_summary)

#     # Step 5: executive summary
#     executive = generate_executive_summary(section_summaries, mode=mode)

#     # -------------------------------
#     # NEW STEP: Knowledge Graph
#     # -------------------------------

#     relation_extractor = RelationExtractor()
#     graph_builder = GraphBuilder()

#     all_triples = []

#     for section in section_summaries:

#         entities_dict = extract_entities(section["summary"], mode)

#         entity_list = flatten_entities(entities_dict)

#         triples = relation_extractor.extract_relations(
#             section["summary"],
#             entity_list
#         )

#         all_triples.extend(triples)

#     if all_triples:
#         graph_builder.insert_triples(all_triples)

#     # -------------------------------

#     return {
#         "sections": section_summaries,
#         "executive_summary": executive,
#         "triples_created": len(all_triples)
#     }

from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary

from v2.pipelines.entity_extraction.entity_extractor import extract_entities

from v2.graph.relation_extractor import RelationExtractor
from v2.graph.graph_builder import GraphBuilder
from v2.graph.entity_utils import flatten_entities


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
    # (SECTION-BASED to reduce LLM calls)
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

        print("Entities:", entity_list)

        if not entity_list:
            continue

        triples = relation_extractor.extract_relations(
            section_text,
            entity_list
        )

        print("Triples:", triples)

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

        print(f"Triples generated: {len(unique_triples)}")

        graph_builder.insert_triples(unique_triples)

    else:

        print("⚠️ No triples generated from document")

    # -----------------------------------------
    # Return pipeline output
    # -----------------------------------------

    return {
        "sections": section_summaries,
        "executive_summary": executive,
        "triples_created": len(all_triples)
    }