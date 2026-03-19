from pathlib import Path
import tempfile

import fitz  # PyMuPDF
import pytesseract
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="PDF Text Extractor with OCR")

# Change this path if your Tesseract is installed somewhere else
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_pdf_text_with_ocr(file_path: str) -> str:
    ocr_parts = []

    doc = fitz.open(file_path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img).strip()
            if page_text:
                ocr_parts.append(page_text)
    finally:
        doc.close()

    return "\n\n".join(ocr_parts).strip()


def extract_pdf_text(file_path: str) -> tuple[str, list[str], int, str]:
    warnings = []
    text_parts = []

    doc = fitz.open(file_path)
    try:
        page_count = len(doc)

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
            else:
                warnings.append(f"No extractable text found on page {page_number}.")
    finally:
        doc.close()

    extracted_text = "\n\n".join(text_parts).strip()

    if extracted_text:
        return extracted_text, warnings, page_count, "Direct PDF text"

    warnings.append("No embedded text found. OCR fallback used.")
    ocr_text = extract_pdf_text_with_ocr(file_path)

    if not ocr_text:
        warnings.append("OCR also found no text.")
    else:
        warnings.append("Text extracted with OCR.")

    return ocr_text, warnings, page_count, "OCR"


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        extracted_text, warnings, page_count, method = extract_pdf_text(temp_path)

        return {
            "success": True,
            "file_name": file.filename,
            "page_count": page_count,
            "character_count": len(extracted_text),
            "method": method,
            "warnings": warnings,
            "extracted_text": extracted_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    print("\nPDF Text Extractor is starting...")
    print(f"Open this in your browser: http://{host}:{port}\n")

    uvicorn.run("pdf_web.main:app", host=host, port=port, reload=False)