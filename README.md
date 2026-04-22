# PDF/Image Text Extractor + ESG Scope 1-3 AI Analysis

Upload multiple PDF and image files and extract text using:

- PyMuPDF (fast direct extraction)
- Tesseract OCR (fallback for scanned PDFs/images)
- Batch processing endpoint for multiple files
- GPT-based Scope 1/2/3 extraction for key points and values
- Word (`.docx`) summary report download

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env
python -m pdf_web.main
```

## Environment variables

Use a local `.env` file (this is ignored by git):

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```
