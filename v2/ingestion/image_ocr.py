"""
Image support 
"""
import fitz
import pytesseract
from PIL import Image
import io
from v2.config import settings

pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

def extract_image_text(pdf_path):

    doc = fitz.open(pdf_path)

    ocr_results = []

    for page_index in range(len(doc)):

        page = doc.load_page(page_index)

        images = page.get_images(full=True)

        for img in images:

            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))

            text = pytesseract.image_to_string(image)

            if text.strip():
                ocr_results.append(text)

    return ocr_results