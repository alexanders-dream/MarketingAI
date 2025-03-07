import json
import logging
import streamlit as st
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader

# Define prompt template for marketing data extraction
logger = logging.getLogger(__name__)

DEFAULT_RESULT = {
    "brand_description": "Not specified",
    "target_audience": "Not specified",
    "products_services": [],
    "marketing_goals": [],
    "existing_content": [],
    "keywords": [],
    "suggested_topics": []
}

qa_prompt = PromptTemplate(
    template="""Extract marketing data as JSON with these EXACT keys:
    {{
        "brand_description": "string",
        "target_audience": "string",
        "products_services": ["string"],
        "marketing_goals": ["string"],
        "existing_content": ["string"],
        "keywords": ["string"],
        "suggested_topics": ["string"]
    }}
    Rules:
    1. Use ONLY document content
    2. Empty values = "Not specified"
    3. Arrays minimum 3 items
    4. Strict valid JSON format
    
    Document: {context}""",
    input_variables=["context"]
)

def parse_extraction_text(text: str) -> Dict[str, Any]:
    try:
        clean_text = text.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_text)
        return {
            **DEFAULT_RESULT,
            **{k: v for k, v in data.items() if k in DEFAULT_RESULT}
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Parsing failed: {str(e)}")
        return DEFAULT_RESULT
    
def extract_data_from_text(llm, file_path, knowledge_base=None):
    """Extract structured marketing data using text-based parsing"""
    
    # Add debug logging
    print(f"Raw extraction text:\n{text}")  # Remove after debugging
    
    # Define default result structure
    default_result = {
        "brand_description": "Not specified",
        "target_audience": "Not specified",
        "products_services": "Not specified",
        "marketing_goals": "Not specified",
        "existing_content": "Not specified",
        "keywords": "Not specified",
        "suggested_topics": []
    }
    
    try:
        if knowledge_base:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=knowledge_base,
                chain_type="stuff"
            )
            # Corrected: Use "query" instead of "input"
            response = qa_chain.invoke({"query": "Extract marketing data"})
            result_text = response["result"]
        else:
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            text = " ".join([doc.page_content for doc in documents])
            response = llm.invoke([{"role": "user", "content": qa_prompt.format(context=text)}])
            result_text = response.content
        
        # Parse the text response
        return parse_extraction_text(result_text)
    
    except Exception as e:
        st.error(f"Data extraction error: {str(e)}")
        # Use predefined default result
        return default_result