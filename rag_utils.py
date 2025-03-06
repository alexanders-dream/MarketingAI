# rag_utils.py
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = get_embedding_model()

@st.cache_resource
def get_chroma_collection(file_id):
    # Use EphemeralClient for in-memory storage, avoiding tenant issues
    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"rag_{file_id}", get_or_create=True)
    return collection

def split_text_into_chunks(text, chunk_size=500):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_chunks(chunks):
    return embedding_model.encode(chunks).tolist()

def add_documents_to_collection(collection, chunks, embeddings):
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )

def retrieve_relevant_chunks(collection, query, n_results=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []