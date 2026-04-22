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


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env(BASE_DIR.parent / ".env")

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


def build_summary_pdf_bytes(title: str, combined_summary: str, results: list[dict]) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    margin = 50
    y = margin
    page_height = page.rect.height
    content_width = page.rect.width - (margin * 2)

    def add_line(text: str, fontsize: int = 11, spacing: float = 16.0, is_bold: bool = False):
        nonlocal page, y
        if y > page_height - margin:
            page = doc.new_page()
            y = margin
        fontname = "helv" if not is_bold else "helv"
        page.insert_text((margin, y), text, fontsize=fontsize, fontname=fontname)
        y += spacing

    def add_paragraph(text: str, fontsize: int = 11, spacing_after: float = 10.0):
        nonlocal page, y
        chunks = re.split(r"(?<=[.!?])\s+", (text or "").strip()) or [""]
        for sentence in chunks:
            if not sentence:
                continue
            rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
            used = page.insert_textbox(rect, sentence, fontsize=fontsize, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
            if used < 0:
                page = doc.new_page()
                y = margin
                rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
                page.insert_textbox(rect, sentence, fontsize=fontsize, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
            y += max(16, fontsize + 6)
        y += spacing_after

    add_line(title or "PDF Extraction Summary", fontsize=16, spacing=24, is_bold=True)
    add_line("Combined Summary", fontsize=13, spacing=18, is_bold=True)
    add_paragraph(combined_summary or "No combined summary generated.")

    if results:
        add_line("Per-file Summaries", fontsize=13, spacing=18, is_bold=True)
        for item in results:
            add_line(f"- {item.get('file_name', 'Unknown file')}", fontsize=11, spacing=14, is_bold=True)
            file_summary = item.get("summary", "")
            add_paragraph(file_summary or "No summary generated.", fontsize=10, spacing_after=8)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _extract_tables_from_pdf_page(page: fitz.Page, page_number: int) -> list[dict]:
    tables = []
    try:
        found_tables = page.find_tables()
    except Exception:
        return tables

    for idx, table in enumerate(found_tables.tables, start=1):
        rows = table.extract()
        cleaned_rows = []
        for row in rows:
            cleaned_rows.append([(cell or "").strip() for cell in row])
        if cleaned_rows:
            tables.append(
                {
                    "source": "pdf",
                    "page": page_number,
                    "table_index": idx,
                    "rows": cleaned_rows,
                }
            )
    return tables


def _extract_structured_rows_from_image(file_path: str) -> list[dict]:
    with Image.open(file_path) as image:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    rows_by_key: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    total_items = len(data.get("text", []))
    for i in range(total_items):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if conf < 35:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        rows_by_key.setdefault(key, []).append((data["left"][i], text))

    rows = []
    for key in sorted(rows_by_key):
        row_tokens = sorted(rows_by_key[key], key=lambda token: token[0])
        if len(row_tokens) < 2:
            continue
        rows.append([token[1] for token in row_tokens])

    if not rows:
        return []

    return [
        {
            "source": "image",
            "page": 1,
            "table_index": 1,
            "rows": rows,
        }
    ]


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


def extract_pdf_text(file_path: str) -> tuple[str, list[str], int, str, dict]:
    warnings = []
    text_parts = []
    structured_data = {"tables": [], "image_count": 0}

    doc = fitz.open(file_path)
    try:
        page_count = len(doc)

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
            else:
                warnings.append(f"No extractable text found on page {page_number}.")
            structured_data["tables"].extend(_extract_tables_from_pdf_page(page, page_number))
            structured_data["image_count"] += len(page.get_images(full=True))
    finally:
        doc.close()

    extracted_text = "\n\n".join(text_parts).strip()
    if structured_data["tables"]:
        warnings.append(f"Detected {len(structured_data['tables'])} table(s) in PDF.")
    if structured_data["image_count"]:
        warnings.append(f"Detected {structured_data['image_count']} embedded image(s) in PDF.")

    if extracted_text:
        return extracted_text, warnings, page_count, "Direct PDF text", structured_data

    warnings.append("No embedded text found. OCR fallback used.")
    ocr_text = extract_pdf_text_with_ocr(file_path)

    if not ocr_text:
        warnings.append("OCR also found no text.")
    else:
        warnings.append("Text extracted with OCR.")

    return ocr_text, warnings, page_count, "OCR", structured_data


def extract_text_from_file(file_path: str, extension: str) -> tuple[str, list[str], int, str, dict]:
    if extension == ".pdf":
        return extract_pdf_text(file_path)

    text = extract_image_text(file_path)
    tables = _extract_structured_rows_from_image(file_path)
    warnings: list[str] = []
    if not text:
        warnings.append("OCR found no text in the image.")
    else:
        warnings.append("Text extracted from image using OCR.")
    if tables:
        warnings.append(f"Detected probable table rows in image ({len(tables)} table block).")
    return text, warnings, 1, "Image OCR", {"tables": tables, "image_count": 1}


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
    api_key = os.getenv("OPENAI_API_KEY")
    input_has_text = bool((text or "").strip())
    heuristic["troubleshooting"] = {
        "used_gpt": False,
        "input_has_text": input_has_text,
        "reason": "",
    }

    if not input_has_text:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        heuristic["note"] = "No extracted text available. Returned heuristic scope analysis."
        heuristic["troubleshooting"]["reason"] = "No extracted text to analyze."
        return heuristic

    if not api_key:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = None
        heuristic["note"] = "OPENAI_API_KEY not configured. Returned heuristic scope analysis."
        heuristic["troubleshooting"]["reason"] = "OPENAI_API_KEY not configured."
        return heuristic

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
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
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text[:120000]},
            ],
            temperature=0,
        )
        raw = response.output_text.strip()
        data = json.loads(raw)
        data["analysis_method"] = "gpt"
        data["model"] = model_name
        data["troubleshooting"] = {
            "used_gpt": True,
            "input_has_text": input_has_text,
            "reason": "",
        }
        return data
    except Exception as exc:
        heuristic["analysis_method"] = "heuristic_fallback"
        heuristic["model"] = model_name
        heuristic["note"] = "GPT analysis failed. Returned heuristic scope analysis."
        heuristic["troubleshooting"] = {
            "used_gpt": False,
            "input_has_text": input_has_text,
            "reason": f"GPT analysis failed: {exc}",
        }
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

        extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(temp_path, extension)

        return {
            "success": True,
            "file_name": file.filename,
            "page_count": page_count,
            "character_count": len(extracted_text),
            "method": method,
            "warnings": warnings,
            "structured_data": structured_data,
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

            extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(temp_path, extension)
            summary = generate_summary(extracted_text)

            extraction_results.append(
                {
                    "success": True,
                    "file_name": file.filename,
                    "page_count": page_count,
                    "character_count": len(extracted_text),
                    "method": method,
                    "warnings": warnings,
                    "structured_data": structured_data,
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

            extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(temp_path, extension)
            extraction_results.append(
                {
                    "file_name": file.filename,
                    "page_count": page_count,
                    "method": method,
                    "warnings": warnings,
                    "structured_data": structured_data,
                    "character_count": len(extracted_text),
                }
            )
            if extracted_text:
                combined_text_parts.append(extracted_text)
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
    }


@app.post("/download-summary-pdf")
def download_summary_pdf(payload: SummaryRequest):
    pdf_bytes = build_summary_pdf_bytes(
        title=payload.title or "PDF Extraction Summary",
        combined_summary=payload.combined_summary,
        results=payload.results,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = Path(temp_file.name)
        temp_path.write_bytes(pdf_bytes)

    return FileResponse(
        path=temp_path,
        media_type="application/pdf",
        filename="pdf_summary_report.pdf",
        background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
    )


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    print("\nPDF Text Extractor is starting...")
    print(f"Open this in your browser: http://{host}:{port}\n")

    uvicorn.run("pdf_web.main:app", host=host, port=port, reload=False)
