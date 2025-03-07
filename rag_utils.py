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
@st.cache_resource
def get_knowledge_base(file_id, file_path):
    """Create FAISS knowledge base with marketing-specific settings"""
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  # Increased for better context
        chunk_overlap=300,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings with enhanced parameters
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return FAISS.from_documents(text_chunks, embeddings).as_retriever(
        search_kwargs={"k": 5}  # Return more relevant chunks
    )