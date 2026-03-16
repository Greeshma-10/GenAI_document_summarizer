from v2.pipelines.summarization.chunking import chunk_text
from v2.pipelines.summarization.summarizer import summarize_chunks
from v2.pipelines.summarization.semantic_section_builder import build_semantic_sections
from v2.pipelines.summarization.section_summarizer import summarize_section
from v2.pipelines.summarization.executive_summarizer import generate_executive_summary


def run_summarization_pipeline(text: str, mode: str = "research"):

    # Step 1: chunk document
    chunks = chunk_text(text)

    # Step 2: summarize chunks
    chunk_summaries = summarize_chunks(chunks, mode=mode)

    # Step 3: semantic clustering
    sections = build_semantic_sections(chunk_summaries, mode=mode)

    # Step 4: summarize sections
    section_summaries = []

    for section in sections:

        section_summary = summarize_section(
            section["section_chunks"],
            section["section_id"]
        )

        section_summaries.append(section_summary)

    # Step 5: executive summary
    executive = generate_executive_summary(section_summaries, mode=mode)

    return {
        "sections": section_summaries,
        "executive_summary": executive
    }