# PDF Text Extractor with OCR + Batch Summary Export

Upload one or multiple PDF files and extract text using:

- PyMuPDF (fast direct extraction)
- Tesseract OCR (fallback for scanned PDFs)
- Batch processing endpoint for multiple PDFs
- Word (`.docx`) summary report download

## Run locally

```bash
pip install -r requirements.txt
python -m pdf_web.main
