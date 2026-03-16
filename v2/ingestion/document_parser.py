from v2.ingestion.text_extractor import extract_text_from_pdf
from v2.ingestion.image_ocr import extract_image_text


def parse_document(file_path):

    text = extract_text_from_pdf(file_path)

    image_text = extract_image_text(file_path)

    return {
        "text": text,
        "images": image_text
    }


def build_document_text(parsed_doc):

    combined_text = parsed_doc["text"]

    for img_text in parsed_doc["images"]:
        combined_text += "\n\nIMAGE_TEXT:\n"
        combined_text += img_text

    return combined_text