from fastapi import UploadFile, HTTPException  # type: ignore
from typing import Dict
import os
import tempfile
import shutil
import re
import fitz  # type: ignore 

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# CLEANING LAYER

def clean_text(text: str) -> str:

    # Remove common watermark patterns
    text = re.sub(r"Vtucircle\.com", "", text, flags=re.IGNORECASE)

    # Remove page numbers like "Page 4"
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)

    # Remove common academic module headers
    text = re.sub(r"BigDataAnalytics-[A-Za-z0-9\-]+", "", text)

    # Replace bullet symbol with dash
    text = re.sub(r"•", "-", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

# INGESTION FUNCTION

def ingest_document(file: UploadFile) -> Dict[str, str]:

    # Validate uploaded file
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename.strip()

    if filename == "":
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Validate extension
    _, extension = os.path.splitext(filename)

    if not extension:
        raise HTTPException(status_code=400, detail="File must have an extension")

    extension = extension.lower().lstrip('.')

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            file.file.seek(0)
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
    except Exception:
        raise HTTPException(status_code=500, detail="Error saving the uploaded file")

    print("temp_path:", temp_path)
    print("File exists:", os.path.exists(temp_path))
    print("File size:", os.path.getsize(temp_path))

    # Extract text
    raw_text = extract_text(temp_path, extension)

    # CLEAN TEXT 
    cleaned_text = clean_text(raw_text)

    return {
        "document_name": filename,
        "text": cleaned_text
    }

# TEXT EXTRACTION

def extract_text(temp_path: str, extension: str) -> str:
    try:
        if extension == "pdf":
            text_chunks = []

            with fitz.open(temp_path) as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text:
                        text_chunks.append(page_text)

            return "\n\n".join(text_chunks).strip()

        elif extension == "txt":
            with open(temp_path, "r", encoding="utf-8") as f:
                return f.read().strip()

        else:
            raise ValueError(f"Unsupported file type: {extension}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
