from io import BytesIO

import pdfplumber
from docx import Document


def extract_document_text(uploaded_file):
    if uploaded_file is None:
        return ""

    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    data = uploaded_file.getvalue()

    if suffix == "pdf":
        with pdfplumber.open(BytesIO(data)) as pdf:
            return "\n".join(
                page_text
                for page in pdf.pages
                if (page_text := page.extract_text())
            ).strip()

    if suffix == "docx":
        document = Document(BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    if suffix == "txt":
        return data.decode("utf-8", errors="replace").strip()

    raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")

