# PDF Text Extractor with OCR + Batch Summary Export

Upload one or multiple PDF files and extract text using:

- PyMuPDF (fast direct extraction)
- Tesseract OCR (fallback for scanned PDFs)
- Batch processing endpoint for multiple PDFs
- Word (`.docx`) summary report download
- Single-PDF Scope 1/2/3 ESG-focused analysis (material lines + metric candidates)

## Run locally

```bash
pip install -r requirements.txt
python -m pdf_web.main
