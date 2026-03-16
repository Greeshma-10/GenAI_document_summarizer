import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\greeshmav\Tesseract-OCR\tesseract.exe"

image_path = "extracted_page1_1.png"

text = pytesseract.image_to_string(Image.open(image_path))

print("\nOCR OUTPUT:\n")
print(text)