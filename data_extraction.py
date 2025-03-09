# data_extraction.py (optimized)
import json
import logging
import streamlit as st
from typing import Dict, Any, Optional, Union, List
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader

# Configure logging
logger = logging.getLogger(__name__)

# Configure logging to display logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a handler to display logs in the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Define the default result structure with correct data types
DEFAULT_RESULT = {
    "brand_description": "Not specified",
    "target_audience": "Not specified",
    "products_services": [],
    "marketing_goals": [],
    "existing_content": [],
    "keywords": [],
    "suggested_topics": []
}

# Define individual prompt templates for each marketing data field
PROMPTS = {
    "brand_description": PromptTemplate(
        template="What is the brand description of the company according to the document? "
                "Include key mission, vision, and values. Document: {context}",
        input_variables=["context"]
    ),
    "target_audience": PromptTemplate(
        template="Who is the target audience of the company according to the document? "
                "Include demographics and psychographics. Document: {context}",
        input_variables=["context"]
    ),
    "products_services": PromptTemplate(
        template="What are the main products or services offered by the company? "
                "List at least 3 items. Document: {context}",
        input_variables=["context"]
    ),
    "marketing_goals": PromptTemplate(
        template="What are the marketing goals of the company? "
                "List at least 3 goals. Document: {context}",
        input_variables=["context"]
    ),
    "existing_content": PromptTemplate(
        template="What existing marketing content does the company have? "
                "List at least 3 types of content. Document: {context}",
        input_variables=["context"]
    ),
    "keywords": PromptTemplate(
        template="What are the main keywords related to the company's marketing? "
                "List at least 5 keywords. Document: {context}",
        input_variables=["context"]
    ),
    "suggested_topics": PromptTemplate(
        template="What topics are suggested for the company's marketing content? "
                "List at least 3 topics. Context: {context}",
        input_variables=["context"]
   )
}

def extract_single_field(llm: Any, context: str, prompt_template: PromptTemplate) -> str:
    """Extract a single field using the provided prompt template."""
    try:
        if not llm:
            raise ValueError("Language model is not initialized")
        
        prompt = prompt_template.format(context=context)
        response = llm.invoke([{"role": "user", "content": prompt}])
        result_text = response.content.strip()

        if not result_text:
            logger.warning(f"Empty response for field extraction using prompt: {prompt_template.template}")
            return ""
            
        logger.debug(f"Raw response for field: {result_text}")
        return result_text

    except Exception as e:
        logger.error(f"Error extracting field: {str(e)}", exc_info=True)
        return ""

def parse_single_response(text: str, expected_type: type) -> Any:
    """Parse the AI's response into the expected type."""
    try:
        if expected_type == str:
            return text.strip() if text else "Not specified"
        elif expected_type == list:
            if not text:
                return []
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse list from text, treating as single item")
                return [text.strip()] if text.strip() else []
        else:
            logger.error(f"Unsupported expected_type: {expected_type}")
            return "Not specified" if expected_type == str else []
            
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}", exc_info=True)
        return "Not specified" if expected_type == str else []

def extract_data_from_text(llm: Any, file_path: str, knowledge_base: Optional[Any] = None) -> Dict[str, Any]:
    """Extract structured marketing data using individual prompts for each field."""
    try:
        if not llm or not file_path:
            raise ValueError("Required parameters not provided")
        
        # Load document content
        if knowledge_base:
            # Use knowledge base content
            text = " ".join([doc.text for doc in knowledge_base.get_relevant_documents(query='')])

        else:
            loader = UnstructuredLoader(file_path)
            documents = loader.load()
            text = " ".join([doc for doc in documents])
        
        logger.debug(f"Document content: {text}")

        # Extract each field individually
        extracted_data = DEFAULT_RESULT.copy()

        # Brand Description
        brand_response = extract_single_field(llm, text, PROMPTS["brand_description"])
        extracted_data["brand_description"] = parse_single_response(brand_response, str)

        # Target Audience
        audience_response = extract_single_field(llm, text, PROMPTS["target_audience"])
        extracted_data["target_audience"] = parse_single_response(audience_response, str)

        # Products/Services
        products_response = extract_single_field(llm, text, PROMPTS["products_services"])
        extracted_data["products_services"] = parse_single_response(products_response, list)

        # Marketing Goals
        goals_response = extract_single_field(llm, text, PROMPTS["marketing_goals"])
        extracted_data["marketing_goals"] = parse_single_response(goals_response, list)

        # Existing Content
        content_response = extract_single_field(llm, text, PROMPTS["existing_content"])
        extracted_data["existing_content"] = parse_single_response(content_response, list)

        # Keywords
        keywords_response = extract_single_field(llm, text, PROMPTS["keywords"])
        extracted_data["keywords"] = parse_single_response(keywords_response, list)

        # Suggested Topics
        topics_response = extract_single_field(llm, text, PROMPTS["suggested_topics"])
        extracted_data["suggested_topics"] = parse_single_response(topics_response, list)

        logger.debug(f"Extracted data: {extracted_data}")
        return extracted_data

    except Exception as e:
        logger.error(f"Data extraction error: {str(e)}", exc_info=True)
        if st:
            st.error(f"Data extraction error: {str(e)}")
        return DEFAULT_RESULT
