# data_extraction.py
import json
import streamlit as st


def extract_data_from_text(llm, text):
    """Extract structured marketing data from text using the AI model."""
    prompt = f"""
    Analyze the following document and extract:
    - Brand description
    - Target audience
    - Products or services
    - Marketing goals
    - Existing content
    - Keywords
    - Suggested topics for social media (3-5 topics)

    Use "Not specified" if information is missing.

    Document: {text}

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