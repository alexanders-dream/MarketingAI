# rag_utils.py
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader

# Define working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_embeddings():
    """Load and cache HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

@st.cache_resource
def get_knowledge_base(file_id, file_path):
    """Create and cache a FAISS-based knowledge base from the uploaded file."""
    # Load the document
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # Adjusted for marketing documents
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(documents)

    # Create FAISS index
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    return knowledge_base.as_retriever()