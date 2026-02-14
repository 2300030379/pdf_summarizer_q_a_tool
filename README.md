# 📄 PDF Summarizer + Q&A Tool

A Streamlit web application that allows users to upload PDF files and perform two main tasks:
- Generate concise summaries of the PDF content using Cohere's language model.
- Generate automatic Q&A pairs or ask custom questions based on the PDF content.

---

## Features

- Upload PDF files (up to 10MB) and preview pages as images.
- Extract text from PDF for analysis.
- Summarize large documents by chunking text for better results.
- Generate automatic questions and answers from the content.
- Ask custom questions to get specific answers from the PDF.
- Download outputs in multiple formats: TXT, DOC, CSV.
- Clean and responsive UI using Streamlit.
- API error handling and progress indicators.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pdf-summarizer-qa-tool.git
   cd pdf-summarizer-qa-tool
   
2.Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.Install dependencies:

pip install -r requirements.txt

4.Setup Streamlit secrets:

Create a .streamlit/secrets.toml file with your Cohere API key:

[cohere]

api_key = "your-cohere-api-key"

---
## Usage

Run the Streamlit app:

streamlit run streamlit_app.py

- Upload your PDF file via the uploader.

- Choose to either summarize the document or use the Q&A features.

- Generate summaries or Q&A and view them side-by-side with a PDF preview.

- Download results in TXT, DOC, or CSV format.

## Technologies Used

Streamlit — Web app framework for data apps.

Cohere API — Language models for text summarization and Q&A.

PyPDF2 — PDF text extraction.

pdfplumber — PDF rendering for preview images.

Pillow (PIL) — Image processing.

Python 3.8+

---

## Notes & Limitations

- PDF text extraction quality depends on the PDF content (scanned PDFs might not work well).

- Rate limits on the Cohere API apply; time.sleep is used to respect this.

- Large PDFs are truncated to the first 100,000 characters for processing.

- Summarization is chunk-based for better performance but might still lose some context, we working on it, for better outcome
  
