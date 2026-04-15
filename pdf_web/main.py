from collections import Counter
import io
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


SCOPE_KEYWORDS = {
    "scope_1": [
        "scope 1", "scope i", "direct emissions", "stationary combustion",
        "mobile combustion", "fugitive emissions", "onsite fuel",
    ],
    "scope_2": [
        "scope 2", "scope ii", "indirect emissions", "purchased electricity",
        "purchased steam", "purchased heat", "market-based", "location-based",
    ],
    "scope_3": [
        "scope 3", "scope iii", "value chain emissions", "upstream", "downstream",
        "purchased goods and services", "business travel", "employee commuting",
        "waste generated", "capital goods", "use of sold products",
    ],
}

MATERIAL_KEYWORDS = [
    "ghg", "co2", "co2e", "emissions", "energy", "electricity", "renewable",
    "target", "baseline", "reduction", "intensity", "tco2e", "decarbonization",
    "science based targets", "sbti", "net zero", "carbon neutral",
]


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


def _extract_metric_matches(text: str) -> list[dict]:
    metric_pattern = re.compile(
        r"(?P<label>[A-Za-z][A-Za-z0-9\s\-/()]{0,80})[:\s]+(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>tco2e|co2e|t co2e|mtco2e|kwh|mwh|gwh|%)",
        re.IGNORECASE,
    )
    matches = []
    for match in metric_pattern.finditer(text):
        label = " ".join(match.group("label").split())
        matches.append(
            {
                "label": label[-90:].strip(),
                "value": match.group("value"),
                "unit": match.group("unit"),
                "snippet": text[max(0, match.start() - 80): match.end() + 80].strip(),
            }
        )
    return matches[:30]


def analyze_scope_data(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lowered_lines = [line.lower() for line in lines]

    scope_hits = {"scope_1": [], "scope_2": [], "scope_3": []}
    for index, lower_line in enumerate(lowered_lines):
        original_line = lines[index]
        for scope, keywords in SCOPE_KEYWORDS.items():
            if any(keyword in lower_line for keyword in keywords):
                scope_hits[scope].append(original_line)

    material_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in MATERIAL_KEYWORDS):
            material_lines.append(line)
    material_lines = material_lines[:20]

    metric_candidates = _extract_metric_matches(text)
    metrics_by_scope = {"scope_1": [], "scope_2": [], "scope_3": [], "other": []}

    for metric in metric_candidates:
        snippet = metric["snippet"].lower()
        placed = False
        for scope, keywords in SCOPE_KEYWORDS.items():
            if any(keyword in snippet for keyword in keywords):
                metrics_by_scope[scope].append(metric)
                placed = True
                break
        if not placed:
            metrics_by_scope["other"].append(metric)

    scope_presence = {
        scope: {
            "found": bool(items),
            "mentions": len(items),
            "sample_lines": items[:5],
            "metrics": metrics_by_scope.get(scope, [])[:6],
        }
        for scope, items in scope_hits.items()
    }

    completeness_score = sum(1 for scope in scope_presence.values() if scope["found"]) / 3

    return {
        "scope_presence": scope_presence,
        "material_data_points": material_lines,
        "metrics": metrics_by_scope,
        "recommended_next_steps": [
            "Verify whether Scope 1, 2, and 3 values are reported for the same reporting year.",
            "Capture base year, target year, and reduction percentage for each target statement.",
            "Validate units (tCO2e, kWh, MWh, %) and consolidate duplicates before customer reporting.",
            "Flag missing scope disclosures and request supporting notes for estimation methods.",
        ],
        "completeness_score": round(completeness_score, 2),
    }


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


@app.post("/extract-texts")
async def extract_texts(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extraction_results = []
    combined_text_parts = []

    for file in files:
        if not file.filename:
            continue
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Invalid file: {file.filename}")

        temp_path = None
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            extracted_text, warnings, page_count, method = extract_pdf_text(temp_path)
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
async def analyze_esg_scope(file: UploadFile = File(...)):
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
        quick_summary = generate_summary(extracted_text, max_sentences=6)
        scope_analysis = analyze_scope_data(extracted_text)

        return {
            "success": True,
            "file_name": file.filename,
            "page_count": page_count,
            "character_count": len(extracted_text),
            "method": method,
            "warnings": warnings,
            "quick_summary": quick_summary,
            "scope_analysis": scope_analysis,
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
