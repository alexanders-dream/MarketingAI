import magic
from typing import Union
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st

ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx"
}

def validate_file(file) -> bool:
    try:
        file_type = magic.from_buffer(file.getvalue(), mime=True)
        return (
            file_type in ALLOWED_MIME_TYPES and
            file.size <= 5 * 1024 * 1024
        )
    except Exception as e:
        st.error(f"File validation error: {str(e)}")
        return False

# Rest of extraction functions remain similar
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PdfReader(uploaded_file)
        return "".join(page.extract_text() for page in pdf_reader.pages)
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join(para.text for para in doc.paragraphs)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")
    st.error("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    return ""