# file_handling.py
from PyPDF2 import PdfReader
from docx import Document
import streamlit as st

def validate_file(file):
    if file.size > 5 * 1024 * 1024:  # 5MB
        return False
    return file.name.split('.')[-1] in ['pdf', 'docx', 'txt']

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