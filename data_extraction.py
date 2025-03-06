# data_extraction.py
import json
import streamlit as st
from rag_utils import get_chroma_collection, retrieve_relevant_chunks

def extract_data_from_text(llm, text, file_id=None):
    """Extract structured marketing data from text using RAG and the AI model."""
    # If file_id is provided, use RAG to retrieve relevant chunks
    retrieved_context = ""
    if file_id:
        collection = get_chroma_collection(file_id)
        # Define queries for each data field to retrieve relevant chunks
        queries = {
            "brand_description": "What is the brand description?",
            "target_audience": "Who is the target audience?",
            "products_services": "What are the products or services?",
            "marketing_goals": "What are the marketing goals?",
            "existing_content": "What existing content is mentioned?",
            "keywords": "What keywords are relevant?",
            "suggested_topics": "What topics could be used for social media?"
        }
        retrieved_chunks = []
        for field, query in queries.items():
            chunks = retrieve_relevant_chunks(collection, query, n_results=2)
            retrieved_chunks.extend(chunks)
        retrieved_context = "\n\n".join(set(retrieved_chunks))  # Remove duplicates

    prompt = f"""
    Analyze the following relevant information extracted from a document and extract:
    - Brand description
    - Target audience
    - Products or services
    - Marketing goals
    - Existing content
    - Keywords
    - Suggested topics for social media (3-5 topics)

    Use "Not specified" if information is missing.

    Relevant Information:
    {retrieved_context if retrieved_context else text}

    Output as JSON:
    {{
        "brand_description": "...",
        "target_audience": "...",
        "products_services": "...",
        "marketing_goals": "...",
        "existing_content": "...",
        "keywords": "...",
        "suggested_topics": ["...", "...", "..."]
    }}
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return json.loads(response.content)
    except json.JSONDecodeError:
        st.error("Failed to parse extracted data.")
        return {}
    except Exception as e:
        st.error(f"Error: {e}")
        return {}