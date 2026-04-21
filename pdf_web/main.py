from collections import Counter
import io
import json
import os
import re
from pathlib import Path
import tempfile
import zipfile

import fitz  # PyMuPDF
import pytesseract
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="PDF Text Extractor with OCR")

# Change this path if your Tesseract is installed somewhere else
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class SummaryRequest(BaseModel):
    title: str = Field(default="PDF Extraction Summary")
    combined_summary: str = Field(default="")
    results: list[dict] = Field(default_factory=list)


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_docx_bytes(paragraphs: list[str]) -> bytes:
    paragraph_xml = []
    for paragraph in paragraphs:
        cleaned = _xml_escape(paragraph or "")
        paragraph_xml.append(
            f"<w:p><w:r><w:t xml:space=\"preserve\">{cleaned}</w:t></w:r></w:p>"
        )

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>
    {''.join(paragraph_xml)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


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


def extract_image_text(file_path: str) -> str:
    with Image.open(file_path) as image:
        return pytesseract.image_to_string(image).strip()


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


def extract_text_from_file(file_path: str, extension: str) -> tuple[str, list[str], int, str]:
    if extension == ".pdf":
        return extract_pdf_text(file_path)

    text = extract_image_text(file_path)
    warnings: list[str] = []
    if not text:
        warnings.append("OCR found no text in the image.")
    else:
        warnings.append("Text extracted from image using OCR.")
    return text, warnings, 1, "Image OCR"


def generate_summary(text: str, max_sentences: int = 5) -> str:
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if not cleaned_text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "to", "from",
        "for", "with", "in", "on", "at", "by", "of", "is", "are", "was", "were",
        "be", "been", "being", "this", "that", "these", "those", "it", "its", "as",
        "we", "you", "they", "their", "our", "your", "can", "will", "would", "may",
        "should", "must", "not", "no", "yes", "than", "also", "such", "into", "about",
    }

    words = re.findall(r"[a-zA-Z0-9']+", cleaned_text.lower())
    word_freq = Counter(word for word in words if len(word) > 2 and word not in stop_words)

    if not word_freq:
        return " ".join(sentences[:max_sentences])

    sentence_scores: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence_words = re.findall(r"[a-zA-Z0-9']+", sentence.lower())
        if not sentence_words:
            continue
        score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
        sentence_scores.append((score, sentence))

    top_sentences = sorted(sentence_scores, key=lambda item: item[0], reverse=True)[:max_sentences]
    selected = {sentence for _, sentence in top_sentences}
    ordered_selected = [sentence for sentence in sentences if sentence in selected]
    return " ".join(ordered_selected)


def analyze_scope_data(text: str) -> dict:
    normalized = text or ""
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", normalized)))

    number_pattern = re.compile(
        r"(scope\s*[1-3][^.\n:]*[:\-]?\s*)([\d,]+(?:\.\d+)?)\s*(tco2e|mtco2e|kgco2e|co2e)?",
        flags=re.IGNORECASE,
    )
    metrics = []
    for match in number_pattern.finditer(normalized):
        metrics.append(
            {
                "label": match.group(1).strip(),
                "value": match.group(2).replace(",", ""),
                "unit": (match.group(3) or "").lower(),
                "raw": match.group(0).strip(),
            }
        )

    target_statements = re.findall(
        r"([^.]*\b(target|reduce|reduction|net[- ]zero|decarboni[sz]ation)\b[^.]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    target_lines = [statement[0].strip() for statement in target_statements if statement[0].strip()]

    def scope_presence(scope_label: str) -> dict:
        scope_pattern = scope_label.replace(" ", r"\s*")
        pattern = re.compile(rf"\b{scope_pattern}\b", re.IGNORECASE)
        found = bool(pattern.search(normalized))
        return {"found": found}

    return {
        "scope_presence": {
            "scope_1": scope_presence("scope 1"),
            "scope_2": scope_presence("scope 2"),
            "scope_3": scope_presence("scope 3"),
        },
        "reporting_years": years,
        "metrics": metrics[:30],
        "target_statements": target_lines[:20],
    }


def analyze_scope_data_with_gpt(text: str) -> dict:
    heuristic = analyze_scope_data(text)
    diagnostics = {
        "requested_gpt_analysis": True,
        "api_key_configured": False,
        "input_has_text": bool((text or "").strip()),
        "used_gpt": False,
        "status": "fallback",
        "reason": "",
    }
    api_key = os.getenv("OPENAI_API_KEY")
    diagnostics["api_key_configured"] = bool(api_key)

    if not diagnostics["input_has_text"]:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        diagnostics["reason"] = "No extracted text was available to send to GPT."
        heuristic["troubleshooting"] = diagnostics
        return heuristic

    if not api_key:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        heuristic["note"] = "OPENAI_API_KEY not configured. Returned heuristic scope analysis."
        diagnostics["reason"] = "OPENAI_API_KEY is missing."
        heuristic["troubleshooting"] = diagnostics
        return heuristic

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
    except Exception as exc:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        diagnostics["reason"] = f"OpenAI client initialization failed: {exc}"
        heuristic["troubleshooting"] = diagnostics
        return heuristic

    prompt = """
You are an ESG analyst. Extract key Scope 1, Scope 2, and Scope 3 reporting points and values.
Return strict JSON with this schema:
{
  "scope_presence": {"scope_1": {"found": bool}, "scope_2": {"found": bool}, "scope_3": {"found": bool}},
  "reporting_years": [string],
  "metrics": [{"scope": "scope_1|scope_2|scope_3|unknown", "category": string, "value": string, "unit": string, "year": string, "source_snippet": string}],
  "target_statements": [string],
  "important_points": [string]
}
If data is missing, return empty arrays and found=false.
"""
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:120000]},
            ],
            temperature=0,
        )
        raw = response.output_text.strip()
        data = json.loads(raw)
        data["analysis_method"] = "gpt"
        data["model"] = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        diagnostics["used_gpt"] = True
        diagnostics["status"] = "connected"
        diagnostics["reason"] = "Document text was successfully sent to GPT and parsed."
        data["troubleshooting"] = diagnostics
        return data
    except Exception as exc:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        diagnostics["reason"] = f"GPT request failed: {exc}"
        heuristic["troubleshooting"] = diagnostics
        return heuristic


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported types: PDF and common image formats.")

    temp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        extracted_text, warnings, page_count, method = extract_text_from_file(temp_path, extension)

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


@app.post("/extract-texts")
async def extract_texts(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extraction_results = []
    combined_text_parts = []

    for file in files:
        if not file.filename:
            continue
        extension = Path(file.filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF and image files are supported. Invalid file: {file.filename}",
            )

        temp_path = None
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method = extract_text_from_file(temp_path, extension)
            summary = generate_summary(extracted_text)

            extraction_results.append(
                {
                    "success": True,
                    "file_name": file.filename,
                    "page_count": page_count,
                    "character_count": len(extracted_text),
                    "method": method,
                    "warnings": warnings,
                    "summary": summary,
                    "extracted_text": extracted_text,
                }
            )

            if extracted_text:
                combined_text_parts.append(extracted_text)
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    combined_text = "\n\n".join(combined_text_parts).strip()
    combined_summary = generate_summary(combined_text, max_sentences=8) if combined_text else ""

    return {
        "success": True,
        "file_count": len(extraction_results),
        "combined_character_count": len(combined_text),
        "combined_summary": combined_summary,
        "results": extraction_results,
    }


@app.post("/analyze-esg-scope")
async def analyze_esg_scope(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extraction_results = []
    combined_text_parts = []
    file_connection_status = []

    for file in files:
        if not file.filename:
            continue
        extension = Path(file.filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF and image files are supported. Invalid file: {file.filename}",
            )

        temp_path = None
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method = extract_text_from_file(temp_path, extension)
            extraction_results.append(
                {
                    "file_name": file.filename,
                    "page_count": page_count,
                    "method": method,
                    "warnings": warnings,
                    "character_count": len(extracted_text),
                }
            )
            if extracted_text:
                combined_text_parts.append(extracted_text)
                file_connection_status.append(
                    {
                        "file_name": file.filename,
                        "included_in_gpt_context": True,
                        "extracted_characters": len(extracted_text),
                        "reason": "Text extracted successfully and queued for GPT analysis.",
                    }
                )
            else:
                file_connection_status.append(
                    {
                        "file_name": file.filename,
                        "included_in_gpt_context": False,
                        "extracted_characters": 0,
                        "reason": "No text extracted from this file, so nothing was sent for GPT context.",
                    }
                )
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    combined_text = "\n\n".join(combined_text_parts).strip()
    ai_analysis = analyze_scope_data_with_gpt(combined_text)
    return {
        "success": True,
        "files": extraction_results,
        "combined_character_count": len(combined_text),
        "analysis": ai_analysis,
        "troubleshooting": {
            "files": file_connection_status,
            "gpt_connection": ai_analysis.get("troubleshooting", {}),
        },
    }


@app.post("/download-summary-docx")
def download_summary_docx(payload: SummaryRequest):
    paragraphs = [payload.title or "PDF Extraction Summary", ""]
    if payload.combined_summary:
        paragraphs.extend(["Combined Summary", payload.combined_summary, ""])

    for item in payload.results:
        file_name = item.get("file_name", "Unknown file")
        paragraphs.append(file_name)
        paragraphs.append(
            f"Pages: {item.get('page_count', '-')} | Characters: {item.get('character_count', '-')} | Method: {item.get('method', '-')}"
        )
        summary = item.get("summary", "")
        if summary:
            paragraphs.append(f"Summary: {summary}")
        paragraphs.append("")

    docx_bytes = build_docx_bytes(paragraphs)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_path = Path(temp_file.name)
        temp_path.write_bytes(docx_bytes)

    return FileResponse(
        path=temp_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="pdf_summary_report.docx",
        background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
    )


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    print("\nPDF Text Extractor is starting...")
    print(f"Open this in your browser: http://{host}:{port}\n")

    uvicorn.run("pdf_web.main:app", host=host, port=port, reload=False)
