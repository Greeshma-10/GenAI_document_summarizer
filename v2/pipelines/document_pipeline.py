from v2.ingestion.document_parser import parse_document, build_document_text
from v2.pipelines.summarization.pipeline import run_summarization_pipeline


def process_document(file_path):

    parsed_doc = parse_document(file_path)

    combined_text = build_document_text(parsed_doc)

    summary = run_summarization_pipeline(combined_text)

    return summary