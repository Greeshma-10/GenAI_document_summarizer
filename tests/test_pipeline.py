from v2.pipelines.document_pipeline import process_document

file_path = "sample.pdf"

result = process_document(file_path)

print("\n----- SUMMARY OUTPUT -----\n")

print(result)