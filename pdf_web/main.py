from collections import Counter
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import time
import traceback
import uuid
from pathlib import Path
import tempfile

import fitz  # PyMuPDF
import pytesseract
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

ELECTRICITY_FACTOR_KG_PER_KWH = 0.233
GAS_FACTOR_KG_PER_KWH = 0.184


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pdf_extractor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def log_step(request_id: str, step: str, **kwargs) -> None:
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{request_id}] {step}" + (f" | {details}" if details else ""))


def log_error(request_id: str, step: str, exc: Exception) -> None:
    logger.error(f"[{request_id}] {step} FAILED | {type(exc).__name__}: {exc}")
    logger.error(traceback.format_exc())


def timed_step(request_id: str, step_name: str):
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            log_step(request_id, f"{step_name}_START")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = round((time.perf_counter() - self.start) * 1000, 2)
            if exc_val is None:
                log_step(request_id, f"{step_name}_END", duration_ms=duration_ms)
            else:
                logger.error(f"[{request_id}] {step_name}_END_WITH_ERROR | duration_ms={duration_ms}")

    return _Timer()


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
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
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup started")
    try:
        static_dir = BASE_DIR / "static"
        logger.info(f"BASE_DIR = {BASE_DIR}")
        logger.info(f"Static directory exists = {static_dir.exists()} | path = {static_dir}")
        logger.info(f"Tesseract configured path = {TESSERACT_PATH}")
        logger.info(f"Tesseract exists = {Path(TESSERACT_PATH).exists()}")
        logger.info(f"OPENAI_API_KEY configured = {bool(os.getenv('OPENAI_API_KEY'))}")
        logger.info(f"OPENAI_MODEL = {os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')}")
    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        logger.error(traceback.format_exc())

    logger.info("Application startup finished")
    yield
    logger.info("Application shutdown")


app = FastAPI(title="PDF Text Extractor with OCR and GPT PDF Summary", lifespan=lifespan)


class SummaryRequest(BaseModel):
    title: str = Field(default="PDF Extraction Summary")
    combined_summary: str = Field(default="")
    results: list[dict] = Field(default_factory=list)


class ScopeSummaryRequest(BaseModel):
    title: str = Field(default="Scope 1-3 Emissions Analysis Report")
    analysis: dict = Field(default_factory=dict)
    results: list[dict] = Field(default_factory=list)


def _safe_float(value) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _extract_usage_dict(response) -> dict:
    usage_obj = getattr(response, "usage", None)
    if not usage_obj:
        return {}

    usage = {}
    for attr in ["input_tokens", "output_tokens", "total_tokens"]:
        val = getattr(usage_obj, attr, None)
        if val is not None:
            usage[attr] = val
    return usage


def _estimate_scope_from_activity(scope_data: dict, scope_key: str) -> dict:
    if not isinstance(scope_data, dict):
        return scope_data

    total_kg = 0.0
    for item in scope_data.get("activity_items", []):
        value = _safe_float(item.get("value"))
        unit = (item.get("unit") or "").lower().strip()
        if value is None:
            continue

        if scope_key == "scope_1" and unit == "kwh":
            total_kg += value * GAS_FACTOR_KG_PER_KWH
        elif scope_key == "scope_2" and unit == "kwh":
            total_kg += value * ELECTRICITY_FACTOR_KG_PER_KWH

    if total_kg > 0:
        scope_data["estimated_emissions_tco2e"] = round(total_kg / 1000.0, 4)
        scope_data["estimated_emissions_possible"] = True
    elif "estimated_emissions_tco2e" not in scope_data:
        scope_data["estimated_emissions_tco2e"] = None

    return scope_data


def _normalise_scope_analysis_schema(data: dict) -> dict:
    if not isinstance(data, dict):
        data = {}

    def default_scope(explanation: str = ""):
        return {
            "reported_emissions_found": False,
            "activity_data_found": False,
            "estimated_emissions_possible": False,
            "explanation": explanation,
            "how_calculated": "",
            "activity_items": [],
            "reported_items": [],
            "estimated_emissions_tco2e": None,
        }

    output = {
        "analysis_method": data.get("analysis_method"),
        "model": data.get("model"),
        "usage": data.get("usage", {}),
        "troubleshooting": data.get("troubleshooting", {}),
        "reporting_years": data.get("reporting_years", []),
        "important_points": data.get("important_points", []),
        "calculation_explanation": data.get("calculation_explanation", []),
        "scope_1": default_scope("No Scope 1 evidence found."),
        "scope_2": default_scope("No Scope 2 evidence found."),
        "scope_3": default_scope("No Scope 3 evidence found."),
    }

    for scope_key in ["scope_1", "scope_2", "scope_3"]:
        incoming = data.get(scope_key, {})
        if isinstance(incoming, dict):
            output[scope_key].update(incoming)

        if not isinstance(output[scope_key].get("activity_items"), list):
            output[scope_key]["activity_items"] = []
        if not isinstance(output[scope_key].get("reported_items"), list):
            output[scope_key]["reported_items"] = []

        output[scope_key]["activity_data_found"] = bool(output[scope_key]["activity_items"]) or bool(
            output[scope_key].get("activity_data_found")
        )
        output[scope_key]["reported_emissions_found"] = bool(output[scope_key]["reported_items"]) or bool(
            output[scope_key].get("reported_emissions_found")
        )
        if output[scope_key]["activity_data_found"]:
            output[scope_key]["estimated_emissions_possible"] = True

        output[scope_key] = _estimate_scope_from_activity(output[scope_key], scope_key)

    if not output["important_points"]:
        output["important_points"] = ["No additional important points were extracted."]

    if not output["calculation_explanation"]:
        output["calculation_explanation"] = [
            "Reported emissions totals were not found in the text.",
            "Activity data was identified for possible downstream emissions estimation.",
        ]

    return output


def build_summary_pdf_bytes(
    title: str,
    combined_summary: str,
    results: list[dict],
    important_points: list[str] | None = None,
    usage: dict | None = None,
) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    margin = 50
    y = margin
    page_height = page.rect.height
    content_width = page.rect.width - (margin * 2)

    def ensure_space(height_needed: float = 24):
        nonlocal page, y
        if y + height_needed > page_height - margin:
            page = doc.new_page()
            y = margin

    def add_line(text: str, fontsize: int = 11, spacing: float = 16.0):
        nonlocal y
        ensure_space(spacing + 4)
        page.insert_text((margin, y), text, fontsize=fontsize, fontname="helv")
        y += spacing

    def add_paragraph(text: str, fontsize: int = 11, spacing_after: float = 10.0):
        nonlocal page, y
        cleaned = (text or "").strip()
        if not cleaned:
            return
        rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
        used = page.insert_textbox(
            rect,
            cleaned,
            fontsize=fontsize,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
        if used < 0:
            page = doc.new_page()
            y = margin
            rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
            page.insert_textbox(
                rect,
                cleaned,
                fontsize=fontsize,
                fontname="helv",
                align=fitz.TEXT_ALIGN_LEFT,
            )
        approx_lines = max(1, len(cleaned) // 90 + 1)
        y += approx_lines * (fontsize + 5) + spacing_after

    add_line(title or "PDF Extraction Summary", fontsize=16, spacing=24)
    add_line("Combined Summary", fontsize=13, spacing=18)
    add_paragraph(combined_summary or "No combined summary generated.", fontsize=11, spacing_after=10)

    if important_points:
        add_line("Important Points", fontsize=13, spacing=18)
        for point in important_points:
            add_paragraph(f"• {point}", fontsize=10, spacing_after=4)

    if results:
        add_line("Per-file Summaries", fontsize=13, spacing=18)
        for item in results:
            add_line(f"- {item.get('file_name', 'Unknown file')}", fontsize=11, spacing=14)
            if item.get("document_type"):
                add_paragraph(f"Document type: {item.get('document_type')}", fontsize=10, spacing_after=4)
            add_paragraph(item.get("summary", "") or "No summary generated.", fontsize=10, spacing_after=6)
            key_points = item.get("important_points") or []
            for point in key_points[:5]:
                add_paragraph(f"• {point}", fontsize=9, spacing_after=2)

    if usage:
        add_line("Model Usage", fontsize=13, spacing=18)
        for k, v in usage.items():
            add_paragraph(f"{k}: {v}", fontsize=10, spacing_after=3)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def build_scope_analysis_pdf_bytes(title: str, analysis: dict, results: list[dict]) -> bytes:
    analysis = _normalise_scope_analysis_schema(analysis)

    doc = fitz.open()
    page = doc.new_page()
    margin = 50
    y = margin
    page_height = page.rect.height
    content_width = page.rect.width - (margin * 2)

    def ensure_space(height_needed: float = 24):
        nonlocal page, y
        if y + height_needed > page_height - margin:
            page = doc.new_page()
            y = margin

    def add_line(text: str, fontsize: int = 11, spacing: float = 16.0):
        nonlocal y
        ensure_space(spacing + 4)
        page.insert_text((margin, y), text, fontsize=fontsize, fontname="helv")
        y += spacing

    def add_paragraph(text: str, fontsize: int = 10, spacing_after: float = 8.0):
        nonlocal page, y
        cleaned = (text or "").strip()
        if not cleaned:
            return
        rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
        used = page.insert_textbox(rect, cleaned, fontsize=fontsize, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
        if used < 0:
            page = doc.new_page()
            y = margin
            rect = fitz.Rect(margin, y, margin + content_width, page.rect.height - margin)
            page.insert_textbox(rect, cleaned, fontsize=fontsize, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
        approx_lines = max(1, len(cleaned) // 90 + 1)
        y += approx_lines * (fontsize + 5) + spacing_after

    def render_scope(scope_name: str, scope_data: dict):
        add_line(scope_name.replace("_", " ").title(), fontsize=13, spacing=18)

        add_paragraph(
            f"Reported emissions found: {'Yes' if scope_data.get('reported_emissions_found') else 'No'}",
            fontsize=10,
            spacing_after=3,
        )
        add_paragraph(
            f"Activity data found: {'Yes' if scope_data.get('activity_data_found') else 'No'}",
            fontsize=10,
            spacing_after=3,
        )
        add_paragraph(
            f"Estimated emissions possible: {'Yes' if scope_data.get('estimated_emissions_possible') else 'No'}",
            fontsize=10,
            spacing_after=3,
        )

        estimated = scope_data.get("estimated_emissions_tco2e")
        if estimated is not None:
            add_paragraph(f"Estimated emissions: {estimated} tCO2e", fontsize=10, spacing_after=5)

        if scope_data.get("explanation"):
            add_paragraph(f"Explanation: {scope_data.get('explanation')}", fontsize=10, spacing_after=4)

        if scope_data.get("how_calculated"):
            add_paragraph(f"How calculated: {scope_data.get('how_calculated')}", fontsize=10, spacing_after=6)

        add_paragraph("Activity Items:", fontsize=10, spacing_after=3)
        activity_items = scope_data.get("activity_items", [])
        if activity_items:
            for item in activity_items:
                add_paragraph(
                    f"• {item.get('type', '-')}: {item.get('value', '-')} {item.get('unit', '')} | "
                    f"Year: {item.get('year', '-')} | Source: {item.get('source_excerpt', '')[:150]}",
                    fontsize=9,
                    spacing_after=3,
                )
        else:
            add_paragraph("No activity items extracted.", fontsize=9, spacing_after=4)

        add_paragraph("Reported Emissions Items:", fontsize=10, spacing_after=3)
        reported_items = scope_data.get("reported_items", [])
        if reported_items:
            for item in reported_items:
                add_paragraph(
                    f"• {item.get('type', '-')}: {item.get('value', '-')} {item.get('unit', '')} | "
                    f"Year: {item.get('year', '-')} | Source: {item.get('source_excerpt', '')[:150]}",
                    fontsize=9,
                    spacing_after=3,
                )
        else:
            add_paragraph("No reported emissions items extracted.", fontsize=9, spacing_after=6)

    add_line(title or "Scope 1-3 Emissions Analysis Report", fontsize=16, spacing=24)
    add_paragraph(
        f"Analysis method: {analysis.get('analysis_method', '-')} | Model: {analysis.get('model', '-')}",
        fontsize=10,
        spacing_after=10,
    )

    years = analysis.get("reporting_years", [])
    if years:
        add_line("Reporting Years", fontsize=13, spacing=18)
        add_paragraph(", ".join(years), fontsize=10, spacing_after=8)

    points = analysis.get("important_points", [])
    add_line("Important Points", fontsize=13, spacing=18)
    if points:
        for point in points:
            add_paragraph(f"• {point}", fontsize=10, spacing_after=4)
    else:
        add_paragraph("No additional important points were extracted.", fontsize=10, spacing_after=8)

    render_scope("scope_1", analysis.get("scope_1", {}))
    render_scope("scope_2", analysis.get("scope_2", {}))
    render_scope("scope_3", analysis.get("scope_3", {}))

    calc_explanations = analysis.get("calculation_explanation", [])
    add_line("Calculation Approach", fontsize=13, spacing=18)
    if calc_explanations:
        for explanation in calc_explanations:
            add_paragraph(f"• {explanation}", fontsize=10, spacing_after=4)
    else:
        add_paragraph("No calculation explanation provided.", fontsize=10, spacing_after=8)

    usage = analysis.get("usage", {})
    if usage:
        add_line("Model Usage", fontsize=13, spacing=18)
        for key, value in usage.items():
            add_paragraph(f"{key}: {value}", fontsize=9, spacing_after=3)

    add_line("Documents Analyzed", fontsize=13, spacing=18)
    for item in results:
        add_paragraph(
            f"- {item.get('file_name', 'Unknown')} | pages: {item.get('page_count', '-')} | method: {item.get('method', '-')}",
            fontsize=9,
            spacing_after=2,
        )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _extract_tables_from_pdf_page(page: fitz.Page, page_number: int, request_id: str) -> list[dict]:
    tables = []
    try:
        found_tables = page.find_tables()
    except Exception as exc:
        log_error(request_id, f"TABLE_DETECTION_PAGE_{page_number}", exc)
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


def _extract_structured_rows_from_image(file_path: str, request_id: str) -> list[dict]:
    with timed_step(request_id, "IMAGE_TABLE_ANALYSIS"):
        with Image.open(file_path) as image:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        rows_by_key: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
        total_items = len(data.get("text", []))
        log_step(request_id, "IMAGE_OCR_DATA_PARSED", total_items=total_items)

        for i in range(total_items):
            text = (data["text"][i] or "").strip()
            if not text:
                continue

            conf_raw = str(data["conf"][i])
            conf = int(conf_raw) if conf_raw.lstrip("-").isdigit() else -1
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


def extract_pdf_text_with_ocr(file_path: str, request_id: str) -> str:
    ocr_parts = []
    doc = fitz.open(file_path)

    try:
        total_pages = len(doc)
        log_step(request_id, "PDF_OCR_OPENED", total_pages=total_pages)

        for page_number, page in enumerate(doc, start=1):
            try:
                with timed_step(request_id, f"PDF_OCR_PAGE_{page_number}"):
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img).strip()
                    if page_text:
                        ocr_parts.append(page_text)
                        log_step(request_id, "PDF_OCR_PAGE_TEXT_FOUND", page=page_number, chars=len(page_text))
                    else:
                        log_step(request_id, "PDF_OCR_PAGE_EMPTY", page=page_number)
            except Exception as exc:
                log_error(request_id, f"PDF_OCR_PAGE_{page_number}", exc)
    finally:
        doc.close()

    return "\n\n".join(ocr_parts).strip()


def extract_image_text(file_path: str, request_id: str) -> str:
    with timed_step(request_id, "IMAGE_TEXT_EXTRACTION"):
        with Image.open(file_path) as image:
            text = pytesseract.image_to_string(image).strip()
            log_step(request_id, "IMAGE_TEXT_DONE", chars=len(text))
            return text


def extract_pdf_text(file_path: str, request_id: str) -> tuple[str, list[str], int, str, dict]:
    warnings = []
    text_parts = []
    structured_data = {"tables": [], "image_count": 0}

    with timed_step(request_id, "PDF_DIRECT_TEXT_EXTRACTION"):
        doc = fitz.open(file_path)
        try:
            page_count = len(doc)
            log_step(request_id, "PDF_OPENED", page_count=page_count, file_path=file_path)

            for page_number, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text().strip()
                    if page_text:
                        text_parts.append(page_text)
                        log_step(request_id, "PDF_PAGE_TEXT_FOUND", page=page_number, chars=len(page_text))
                    else:
                        warnings.append(f"No extractable text found on page {page_number}.")
                        log_step(request_id, "PDF_PAGE_NO_TEXT", page=page_number)

                    tables = _extract_tables_from_pdf_page(page, page_number, request_id)
                    structured_data["tables"].extend(tables)
                    structured_data["image_count"] += len(page.get_images(full=True))
                except Exception as exc:
                    log_error(request_id, f"PDF_PAGE_PROCESS_{page_number}", exc)
                    warnings.append(f"Failed to fully process page {page_number}: {exc}")
        finally:
            doc.close()

    extracted_text = "\n\n".join(text_parts).strip()

    if structured_data["tables"]:
        warnings.append(f"Detected {len(structured_data['tables'])} table(s) in PDF.")
    if structured_data["image_count"]:
        warnings.append(f"Detected {structured_data['image_count']} embedded image(s) in PDF.")

    if extracted_text:
        log_step(request_id, "PDF_DIRECT_TEXT_SUCCESS", chars=len(extracted_text))
        return extracted_text, warnings, page_count, "Direct PDF text", structured_data

    warnings.append("No embedded text found. OCR fallback used.")
    log_step(request_id, "PDF_DIRECT_TEXT_EMPTY_USING_OCR")

    ocr_text = extract_pdf_text_with_ocr(file_path, request_id)

    if not ocr_text:
        warnings.append("OCR also found no text.")
        log_step(request_id, "PDF_OCR_EMPTY")
    else:
        warnings.append("Text extracted with OCR.")
        log_step(request_id, "PDF_OCR_SUCCESS", chars=len(ocr_text))

    return ocr_text, warnings, page_count, "OCR", structured_data


def extract_text_from_file(file_path: str, extension: str, request_id: str) -> tuple[str, list[str], int, str, dict]:
    log_step(request_id, "EXTRACT_TEXT_FROM_FILE", file_path=file_path, extension=extension)

    if extension == ".pdf":
        return extract_pdf_text(file_path, request_id)

    text = extract_image_text(file_path, request_id)
    tables = _extract_structured_rows_from_image(file_path, request_id)
    warnings: list[str] = []

    if not text:
        warnings.append("OCR found no text in the image.")
        log_step(request_id, "IMAGE_OCR_NO_TEXT")
    else:
        warnings.append("Text extracted from image using OCR.")
        log_step(request_id, "IMAGE_OCR_SUCCESS", chars=len(text))

    if tables:
        warnings.append(f"Detected probable table rows in image ({len(tables)} table block).")
        log_step(request_id, "IMAGE_TABLES_FOUND", count=len(tables))

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


def _extract_kwh_items(cleaned: str, label: str, type_name: str) -> list[dict]:
    items = []
    pattern = re.compile(
        rf"{label}[^.\n]{{0,160}}?(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*kwh",
        flags=re.IGNORECASE
    )
    for match in pattern.finditer(cleaned):
        value = _safe_float(match.group(1))
        if value is None:
            continue
        excerpt = match.group(0).strip()
        year_match = re.search(r"\b(20\d{2})\b", excerpt)
        year = year_match.group(1) if year_match else ""
        items.append(
            {
                "type": type_name,
                "value": value,
                "unit": "kWh",
                "year": year,
                "source_excerpt": excerpt,
            }
        )
    return items


def _extract_gas_m3_items(cleaned: str) -> list[dict]:
    items = []
    patterns = [
        re.compile(r"consumption[^.\n]{0,80}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*units?\s*\(m3", flags=re.IGNORECASE),
        re.compile(r"gas[^.\n]{0,140}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*m3", flags=re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            value = _safe_float(match.group(1))
            if value is None:
                continue
            excerpt = match.group(0).strip()
            year_match = re.search(r"\b(20\d{2})\b", excerpt)
            year = year_match.group(1) if year_match else ""
            items.append(
                {
                    "type": "natural_gas_m3",
                    "value": value,
                    "unit": "m3",
                    "year": year,
                    "source_excerpt": excerpt,
                }
            )
    return items


def analyze_scope_data(text: str) -> dict:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    years = sorted(set(re.findall(r"\b(20\d{2})\b", cleaned)))

    electricity_items = _extract_kwh_items(cleaned, "(?:electricity|energy used)", "electricity_kwh")
    gas_kwh_items = []
    gas_m3_items = _extract_gas_m3_items(cleaned)

    gas_context_pattern = re.compile(r"gas[^.\n]{0,160}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*kwh", flags=re.IGNORECASE)
    for match in gas_context_pattern.finditer(cleaned):
        value = _safe_float(match.group(1))
        if value is None:
            continue
        excerpt = match.group(0).strip()
        year_match = re.search(r"\b(20\d{2})\b", excerpt)
        gas_kwh_items.append(
            {
                "type": "natural_gas_kwh",
                "value": value,
                "unit": "kWh",
                "year": year_match.group(1) if year_match else "",
                "source_excerpt": excerpt,
            }
        )

    if not gas_kwh_items:
        generic_energy_used = re.finditer(r"energy used[^.\n]{0,80}?(\d+(?:,\d{3})*(?:\.\d+)?)\s*kwh", cleaned, flags=re.IGNORECASE)
        for match in generic_energy_used:
            excerpt = match.group(0).strip()
            nearby = cleaned[max(0, match.start() - 120): match.end() + 40].lower()
            value = _safe_float(match.group(1))
            if value is None:
                continue
            if "gas" in nearby:
                gas_kwh_items.append(
                    {
                        "type": "natural_gas_kwh",
                        "value": value,
                        "unit": "kWh",
                        "year": years[0] if years else "",
                        "source_excerpt": excerpt,
                    }
                )
            elif "electricity" in nearby:
                electricity_items.append(
                    {
                        "type": "electricity_kwh",
                        "value": value,
                        "unit": "kWh",
                        "year": years[0] if years else "",
                        "source_excerpt": excerpt,
                    }
                )

    co2e_pattern = re.compile(
        r"(?P<label>[^.\n]{0,80}?)(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?P<unit>kgco2e|tco2e|co2e|mtco2e)",
        flags=re.IGNORECASE,
    )

    reported_scope_1 = []
    reported_scope_2 = []
    reported_scope_3 = []

    for match in co2e_pattern.finditer(cleaned):
        value = _safe_float(match.group("value"))
        if value is None:
            continue
        unit = match.group("unit")
        excerpt = match.group(0).strip()
        label_text = (match.group("label") or "").lower()

        item = {
            "type": "reported_emissions",
            "value": value,
            "unit": unit,
            "year": years[0] if years else "",
            "source_excerpt": excerpt,
        }

        if "scope 1" in label_text or "gas" in label_text or "fuel" in label_text:
            reported_scope_1.append(item)
        elif "scope 2" in label_text or "electricity" in label_text:
            reported_scope_2.append(item)
        elif "scope 3" in label_text or "travel" in label_text or "waste" in label_text or "supplier" in label_text:
            reported_scope_3.append(item)

    output = {
        "analysis_method": "heuristic_fallback",
        "model": None,
        "usage": {},
        "reporting_years": years,
        "important_points": [],
        "scope_1": {
            "reported_emissions_found": bool(reported_scope_1),
            "activity_data_found": bool(gas_kwh_items or gas_m3_items),
            "estimated_emissions_possible": bool(gas_kwh_items or gas_m3_items),
            "explanation": "Gas consumption is direct fuel use and usually maps to Scope 1." if (gas_kwh_items or gas_m3_items) else "No Scope 1 evidence found.",
            "how_calculated": (
                "Convert gas activity data (kWh or m3) to emissions using an appropriate gas emission factor."
                if (gas_kwh_items or gas_m3_items)
                else ""
            ),
            "activity_items": gas_kwh_items + gas_m3_items,
            "reported_items": reported_scope_1,
            "estimated_emissions_tco2e": None,
        },
        "scope_2": {
            "reported_emissions_found": bool(reported_scope_2),
            "activity_data_found": bool(electricity_items),
            "estimated_emissions_possible": bool(electricity_items),
            "explanation": "Purchased electricity consumption usually maps to Scope 2." if electricity_items else "No Scope 2 evidence found.",
            "how_calculated": "Convert electricity kWh using a market based or location based electricity factor." if electricity_items else "",
            "activity_items": electricity_items,
            "reported_items": reported_scope_2,
            "estimated_emissions_tco2e": None,
        },
        "scope_3": {
            "reported_emissions_found": bool(reported_scope_3),
            "activity_data_found": False,
            "estimated_emissions_possible": False,
            "explanation": "No value chain emissions evidence found in the text.",
            "how_calculated": "",
            "activity_items": [],
            "reported_items": reported_scope_3,
            "estimated_emissions_tco2e": None,
        },
        "calculation_explanation": [
            "Reported emissions totals were not found in the text.",
            "Activity data was identified for possible downstream emissions estimation.",
        ],
        "troubleshooting": {
            "used_gpt": False,
            "input_has_text": bool(cleaned),
            "reason": "Heuristic parser used.",
        },
    }

    output = _normalise_scope_analysis_schema(output)

    important_points = []
    if output["scope_1"]["activity_data_found"]:
        important_points.append("Gas activity data was identified and mapped to Scope 1.")
    if output["scope_2"]["activity_data_found"]:
        important_points.append("Electricity activity data was identified and mapped to Scope 2.")
    if output["scope_3"]["reported_emissions_found"] or output["scope_3"]["activity_data_found"]:
        important_points.append("Possible Scope 3 evidence was identified in the document.")
    if not important_points:
        important_points.append("No direct emissions values were found. Activity data may still be limited.")

    output["important_points"] = important_points
    return output


def analyze_scope_data_with_gpt(text: str) -> dict:
    if not (text or "").strip():
        fallback = analyze_scope_data("")
        fallback["troubleshooting"] = {
            "used_gpt": False,
            "input_has_text": False,
            "reason": "No extracted text available for ESG scope analysis.",
        }
        return fallback

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        fallback = analyze_scope_data(text)
        fallback["troubleshooting"] = {
            "used_gpt": False,
            "input_has_text": True,
            "reason": "OPENAI_API_KEY not configured; used heuristic fallback.",
        }
        return fallback

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = """
You are an ESG emissions analyst.

Analyse the extracted document text and return STRICT JSON only.

Your task is to identify BOTH:
1. reported emissions explicitly stated in the document
2. activity data that can be used to estimate emissions

Return JSON with exactly this structure:

{
  "reporting_years": [],
  "important_points": [],
  "scope_1": {
    "reported_emissions_found": false,
    "activity_data_found": false,
    "estimated_emissions_possible": false,
    "explanation": "",
    "how_calculated": "",
    "activity_items": [
      {
        "type": "",
        "value": 0,
        "unit": "",
        "year": "",
        "source_excerpt": ""
      }
    ],
    "reported_items": [
      {
        "type": "",
        "value": 0,
        "unit": "",
        "year": "",
        "source_excerpt": ""
      }
    ],
    "estimated_emissions_tco2e": null
  },
  "scope_2": {
    "reported_emissions_found": false,
    "activity_data_found": false,
    "estimated_emissions_possible": false,
    "explanation": "",
    "how_calculated": "",
    "activity_items": [],
    "reported_items": [],
    "estimated_emissions_tco2e": null
  },
  "scope_3": {
    "reported_emissions_found": false,
    "activity_data_found": false,
    "estimated_emissions_possible": false,
    "explanation": "",
    "how_calculated": "",
    "activity_items": [],
    "reported_items": [],
    "estimated_emissions_tco2e": null
  },
  "calculation_explanation": []
}

Rules:
- Return valid JSON only.
- Do not invent reported emissions values.
- If the document contains electricity usage in kWh, classify it as Scope 2 activity data.
- If the document contains gas usage in kWh or m3, classify it as Scope 1 activity data.
- Only put values into reported_items if CO2e values are explicitly written in the text.
- Utility bills often contain activity data, not direct emissions totals.
- If activity data is found, set activity_data_found to true even if reported emissions are absent.
- Scope 3 should only be marked if there is real value chain evidence.
- Use concise, business-friendly explanations.
- If there is no evidence for a scope, keep activity_items and reported_items empty.
"""

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": (text or "")[:120000]},
            ],
            text={"format": {"type": "json_object"}},
        )

        parsed = json.loads(response.output_text.strip())
        parsed["analysis_method"] = "gpt"
        parsed["model"] = model_name
        parsed["usage"] = _extract_usage_dict(response)
        parsed["troubleshooting"] = {
            "used_gpt": True,
            "input_has_text": True,
            "reason": "",
        }

        parsed = _normalise_scope_analysis_schema(parsed)

        if not parsed["important_points"]:
            parsed["important_points"] = [
                "Activity data was identified for possible downstream emissions estimation."
            ]

        return parsed

    except Exception as exc:
        fallback = analyze_scope_data(text)
        fallback["troubleshooting"] = {
            "used_gpt": False,
            "input_has_text": True,
            "reason": f"GPT scope analysis failed: {exc}",
        }
        return fallback


def analyse_documents_with_gpt(file_payloads: list[dict], combined_text: str, request_id: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not combined_text.strip():
        log_step(request_id, "GPT_DOCUMENT_ANALYSIS_SKIPPED", reason="No combined text")
        return {
            "title": "Document Analysis Summary",
            "combined_summary": "No extracted text available.",
            "important_points": [],
            "results": [],
            "analysis_method": "empty_text",
            "model": None,
            "usage": {},
            "troubleshooting": {
                "used_gpt": False,
                "reason": "No combined text",
            },
        }

    if not api_key:
        log_step(request_id, "GPT_DOCUMENT_ANALYSIS_SKIPPED", reason="OPENAI_API_KEY not configured")
        fallback_results = []
        for item in file_payloads:
            fallback_results.append(
                {
                    "file_name": item["file_name"],
                    "summary": generate_summary(item.get("extracted_text", "")),
                    "important_points": [],
                    "document_type": "",
                }
            )
        return {
            "title": "Document Analysis Summary",
            "combined_summary": generate_summary(combined_text, max_sentences=8),
            "important_points": [],
            "results": fallback_results,
            "analysis_method": "heuristic_fallback",
            "model": None,
            "usage": {},
            "troubleshooting": {
                "used_gpt": False,
                "reason": "OPENAI_API_KEY not configured",
            },
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        file_descriptions = []
        for item in file_payloads:
            file_descriptions.append(
                {
                    "file_name": item["file_name"],
                    "text": item.get("extracted_text", "")[:40000],
                }
            )

        prompt = """
You are a document analysis assistant.

Analyse the uploaded documents and return STRICT JSON only.

Required JSON schema:
{
  "title": "Document Analysis Summary",
  "combined_summary": "Overall summary across all files",
  "important_points": ["point 1", "point 2", "point 3"],
  "results": [
    {
      "file_name": "name of file",
      "summary": "clear summary of that file",
      "important_points": ["point 1", "point 2"],
      "document_type": "invoice/report/statement/other"
    }
  ]
}

Rules:
1. Return valid JSON only.
2. results length must match input files.
3. Use the exact file_name values provided.
4. Keep summaries concise but useful.
5. If something is unclear, still provide the best possible summary.
"""

        user_payload = {
            "files": file_descriptions,
            "combined_text": combined_text[:120000],
        }

        with timed_step(request_id, "OPENAI_DOCUMENT_ANALYSIS"):
            log_step(
                request_id,
                "OPENAI_REQUEST",
                model=model_name,
                file_count=len(file_payloads),
                input_chars=min(len(combined_text), 120000),
            )

            response = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_object"}},
            )

            raw = response.output_text.strip()
            usage = _extract_usage_dict(response)
            log_step(request_id, "OPENAI_RESPONSE_RECEIVED", output_chars=len(raw), usage=usage)

            data = json.loads(raw)
            data["analysis_method"] = "gpt"
            data["model"] = model_name
            data["usage"] = usage
            data["troubleshooting"] = {
                "used_gpt": True,
                "reason": "",
            }
            return data

    except Exception as exc:
        log_error(request_id, "OPENAI_DOCUMENT_ANALYSIS", exc)

        fallback_results = []
        for item in file_payloads:
            fallback_results.append(
                {
                    "file_name": item["file_name"],
                    "summary": generate_summary(item.get("extracted_text", "")),
                    "important_points": [],
                    "document_type": "",
                }
            )

        return {
            "title": "Document Analysis Summary",
            "combined_summary": generate_summary(combined_text, max_sentences=8),
            "important_points": [],
            "results": fallback_results,
            "analysis_method": "heuristic_fallback",
            "model": model_name,
            "usage": {},
            "troubleshooting": {
                "used_gpt": False,
                "reason": f"GPT analysis failed: {exc}",
            },
        }


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        log_step(
            request_id,
            "REQUEST_STARTED",
            method=request.method,
            path=request.url.path,
            client=getattr(request.client, "host", "unknown"),
        )
        response = await call_next(request)
        duration = round((time.perf_counter() - start) * 1000, 2)
        log_step(
            request_id,
            "REQUEST_FINISHED",
            status_code=response.status_code,
            duration_ms=duration,
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        duration = round((time.perf_counter() - start) * 1000, 2)
        log_error(request_id, f"REQUEST_CRASHED after {duration}ms", exc)
        raise


@app.get("/")
def home(request: Request):
    request_id = getattr(request.state, "request_id", "no-id")
    try:
        file_path = BASE_DIR / "static" / "index.html"
        log_step(request_id, "HOME_ROUTE", file_exists=file_path.exists(), path=file_path)
        return FileResponse(file_path)
    except Exception as exc:
        log_error(request_id, "HOME_ROUTE", exc)
        raise HTTPException(status_code=500, detail="Failed to load homepage.")


@app.get("/health")
def health():
    return {
        "success": True,
        "base_dir": str(BASE_DIR),
        "static_exists": (BASE_DIR / "static").exists(),
        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        "tesseract_exists": Path(pytesseract.pytesseract.tesseract_cmd).exists()
        if pytesseract.pytesseract.tesseract_cmd
        else False,
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "log_file": str(LOG_FILE),
    }


@app.post("/extract-text")
async def extract_text(request: Request, file: UploadFile = File(...)):
    request_id = getattr(request.state, "request_id", "no-id")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Supported types: PDF and common image formats.")

    temp_path = None

    try:
        with timed_step(request_id, "READ_UPLOAD"):
            contents = await file.read()
            log_step(request_id, "UPLOAD_READ", file_name=file.filename, bytes=len(contents))

        with timed_step(request_id, "SAVE_TEMP_FILE"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name
            log_step(request_id, "TEMP_FILE_SAVED", temp_path=temp_path)

        extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(
            temp_path, extension, request_id
        )

        return {
            "success": True,
            "request_id": request_id,
            "file_name": file.filename,
            "page_count": page_count,
            "character_count": len(extracted_text),
            "method": method,
            "warnings": warnings,
            "structured_data": structured_data,
            "extracted_text": extracted_text,
        }

    except Exception as exc:
        log_error(request_id, "EXTRACT_TEXT_ENDPOINT", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
                log_step(request_id, "TEMP_FILE_REMOVED", temp_path=temp_path)
            except Exception as exc:
                log_error(request_id, "TEMP_FILE_REMOVE", exc)


@app.post("/extract-texts")
async def extract_texts(request: Request, files: list[UploadFile] = File(...)):
    request_id = getattr(request.state, "request_id", "no-id")

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
            log_step(request_id, "MULTI_FILE_START", file_name=file.filename, extension=extension)

            contents = await file.read()
            log_step(request_id, "MULTI_FILE_READ", file_name=file.filename, bytes=len(contents))

            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(
                temp_path, extension, request_id
            )

            file_result = {
                "success": True,
                "file_name": file.filename,
                "page_count": page_count,
                "character_count": len(extracted_text),
                "method": method,
                "warnings": warnings,
                "structured_data": structured_data,
                "extracted_text": extracted_text,
            }
            extraction_results.append(file_result)

            if extracted_text:
                combined_text_parts.append(extracted_text)

        except Exception as exc:
            log_error(request_id, f"MULTI_FILE_PROCESS_{file.filename}", exc)
            extraction_results.append(
                {
                    "success": False,
                    "file_name": file.filename,
                    "error": str(exc),
                    "extracted_text": "",
                }
            )

        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                    log_step(request_id, "MULTI_FILE_TEMP_REMOVED", temp_path=temp_path)
                except Exception as exc:
                    log_error(request_id, "MULTI_FILE_TEMP_REMOVE", exc)

    combined_text = "\n\n".join([x for x in combined_text_parts if x]).strip()
    gpt_analysis = analyse_documents_with_gpt(extraction_results, combined_text, request_id)

    merged_results = []
    gpt_by_name = {
        item.get("file_name"): item
        for item in gpt_analysis.get("results", [])
        if isinstance(item, dict)
    }

    for item in extraction_results:
        merged = dict(item)
        gpt_item = gpt_by_name.get(item.get("file_name"), {})
        merged["summary"] = gpt_item.get("summary", "")
        merged["important_points"] = gpt_item.get("important_points", [])
        merged["document_type"] = gpt_item.get("document_type", "")
        merged_results.append(merged)

    return {
        "success": True,
        "request_id": request_id,
        "file_count": len(merged_results),
        "combined_character_count": len(combined_text),
        "combined_summary": gpt_analysis.get("combined_summary", ""),
        "important_points": gpt_analysis.get("important_points", []),
        "results": merged_results,
        "analysis_method": gpt_analysis.get("analysis_method"),
        "model": gpt_analysis.get("model"),
        "usage": gpt_analysis.get("usage", {}),
        "troubleshooting": gpt_analysis.get("troubleshooting", {}),
    }


@app.post("/extract-analyse-and-generate-pdf")
async def extract_analyse_and_generate_pdf(request: Request, files: list[UploadFile] = File(...)):
    request_id = getattr(request.state, "request_id", "no-id")

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
            log_step(request_id, "PIPELINE_FILE_START", file_name=file.filename, extension=extension)

            contents = await file.read()
            log_step(request_id, "PIPELINE_FILE_READ", file_name=file.filename, bytes=len(contents))

            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(
                temp_path, extension, request_id
            )

            file_result = {
                "success": True,
                "file_name": file.filename,
                "page_count": page_count,
                "character_count": len(extracted_text),
                "method": method,
                "warnings": warnings,
                "structured_data": structured_data,
                "extracted_text": extracted_text,
            }
            extraction_results.append(file_result)

            if extracted_text:
                combined_text_parts.append(extracted_text)

        except Exception as exc:
            log_error(request_id, f"PIPELINE_FILE_PROCESS_{file.filename}", exc)
            extraction_results.append(
                {
                    "success": False,
                    "file_name": file.filename,
                    "error": str(exc),
                    "extracted_text": "",
                }
            )

        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                    log_step(request_id, "PIPELINE_TEMP_REMOVED", temp_path=temp_path)
                except Exception as exc:
                    log_error(request_id, "PIPELINE_TEMP_REMOVE", exc)

    combined_text = "\n\n".join([x for x in combined_text_parts if x]).strip()
    gpt_analysis = analyse_documents_with_gpt(extraction_results, combined_text, request_id)

    merged_results = []
    gpt_by_name = {
        item.get("file_name"): item
        for item in gpt_analysis.get("results", [])
        if isinstance(item, dict)
    }

    for item in extraction_results:
        merged = dict(item)
        gpt_item = gpt_by_name.get(item.get("file_name"), {})
        merged["summary"] = gpt_item.get("summary", "")
        merged["important_points"] = gpt_item.get("important_points", [])
        merged["document_type"] = gpt_item.get("document_type", "")
        merged_results.append(merged)

    pdf_bytes = build_summary_pdf_bytes(
        title=gpt_analysis.get("title", "Document Analysis Summary"),
        combined_summary=gpt_analysis.get("combined_summary", ""),
        results=merged_results,
        important_points=gpt_analysis.get("important_points", []),
        usage=gpt_analysis.get("usage", {}),
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = Path(temp_file.name)
        temp_path.write_bytes(pdf_bytes)

    log_step(request_id, "FINAL_PDF_READY", temp_path=temp_path, bytes=len(pdf_bytes))

    return FileResponse(
        path=temp_path,
        media_type="application/pdf",
        filename="document_analysis_report.pdf",
        background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
    )


@app.post("/analyze-esg-scope")
async def analyze_esg_scope(request: Request, files: list[UploadFile] = File(...)):
    request_id = getattr(request.state, "request_id", "no-id")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extraction_results = []
    combined_text_parts = []

    for file in files:
        if not file.filename:
            continue
        extension = Path(file.filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {file.filename}")

        temp_path = None
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method, structured_data = extract_text_from_file(
                temp_path, extension, request_id
            )
            extraction_results.append(
                {
                    "success": True,
                    "file_name": file.filename,
                    "page_count": page_count,
                    "character_count": len(extracted_text),
                    "method": method,
                    "warnings": warnings,
                    "structured_data": structured_data,
                    "extracted_text": extracted_text,
                }
            )
            if extracted_text:
                combined_text_parts.append(extracted_text)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    combined_text = "\n\n".join(combined_text_parts).strip()
    analysis = analyze_scope_data_with_gpt(combined_text)
    analysis["document_count"] = len(extraction_results)

    return {
        "success": True,
        "request_id": request_id,
        "analysis": analysis,
        "results": extraction_results,
    }


@app.post("/download-summary-pdf")
def download_summary_pdf(request: Request, payload: SummaryRequest):
    request_id = getattr(request.state, "request_id", "no-id")

    try:
        with timed_step(request_id, "BUILD_SUMMARY_PDF"):
            pdf_bytes = build_summary_pdf_bytes(
                title=payload.title or "PDF Extraction Summary",
                combined_summary=payload.combined_summary,
                results=payload.results,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_bytes(pdf_bytes)

        log_step(request_id, "SUMMARY_PDF_READY", temp_path=temp_path, bytes=len(pdf_bytes))

        return FileResponse(
            path=temp_path,
            media_type="application/pdf",
            filename="pdf_summary_report.pdf",
            background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
        )

    except Exception as exc:
        log_error(request_id, "DOWNLOAD_SUMMARY_PDF", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate summary PDF: {exc}")


@app.post("/download-scope-summary-pdf")
def download_scope_summary_pdf(request: Request, payload: ScopeSummaryRequest):
    request_id = getattr(request.state, "request_id", "no-id")
    try:
        with timed_step(request_id, "BUILD_SCOPE_SUMMARY_PDF"):
            pdf_bytes = build_scope_analysis_pdf_bytes(
                title=payload.title,
                analysis=payload.analysis,
                results=payload.results,
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_bytes(pdf_bytes)

        return FileResponse(
            path=temp_path,
            media_type="application/pdf",
            filename="scope_1_2_3_analysis_report.pdf",
            background=BackgroundTask(lambda: temp_path.unlink(missing_ok=True)),
        )
    except Exception as exc:
        log_error(request_id, "DOWNLOAD_SCOPE_SUMMARY_PDF", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate scope summary PDF: {exc}")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    print("\nPDF Text Extractor is starting...")
    print(f"Open this in your browser: http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Log file: {LOG_FILE}")
    print(f"Pipeline PDF endpoint: http://{host}:{port}/extract-analyse-and-generate-pdf\n")

    uvicorn.run("pdf_web.main:app", host=host, port=port, reload=False)
