from v2.ingestion.document_parser import parse_document, build_document_text

file_path = "image_test.pdf"

parsed_doc = parse_document(file_path)

combined_text = build_document_text(parsed_doc)

print("\n----- COMBINED DOCUMENT OUTPUT -----\n")
print(combined_text[:1000])