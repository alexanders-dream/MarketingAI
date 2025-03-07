# data_extraction.py
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader

# Define prompt template for marketing data extraction
# data_extraction.py
qa_prompt = PromptTemplate(
    template="""
    Analyze this document and extract SPECIFIC information using EXACTLY these headings:
    
    Brand Description: [Describe the brand's core purpose]
    Target Audience: [Who they serve - include demographics]
    Products/Services: [List offerings with details]
    Marketing Goals: [Clear objectives from the document]
    Existing Content: [Types of content mentioned]
    Keywords: [Important terms separated by commas]
    Suggested Social Media Topics: [3-5 campaign ideas]

    Document Content: {context}

    RULES:
    1. Use ONLY information from the document
    2. Never invent information - use "Not specified" if missing
    3. Maintain the exact heading format shown above
    4. For lists, use bullet points with "-"
    """,
    input_variables=["context"]
)

def parse_extraction_text(text: str) -> dict:
    """Parse the text response into structured data with flexible matching"""

    # Add debug logging
    print(f"Raw extraction text:\n{text}")  # Remove after debugging

    result = {
        "brand_description": "",
        "target_audience": "",
        "products_services": "",
        "marketing_goals": "",
        "existing_content": "",
        "keywords": "",
        "suggested_topics": []
    }

    section_map = {
        "brand description": "brand_description",
        "target audience": "target_audience",
        "products/services": "products_services",
        "marketing goals": "marketing_goals",
        "existing content": "existing_content",
        "keywords": "keywords",
        "suggested social media topics": "suggested_topics"
    }

    current_section = None
    lines = text.split('\n')

    for line in lines:
        line = line.strip().lower()
        # Detect section headers
        for section in section_map:
            if line.startswith(section + ":"):
                current_section = section_map[section]
                value = line.split(":", 1)[-1].strip()
                if value:
                    result[current_section] = value.capitalize()
                break
        else:
            # Handle content lines
            if current_section:
                if line.startswith("-"):
                    clean_line = line[1:].strip()
                    if current_section == "suggested_topics":
                        result[current_section].append(clean_line.capitalize())
                    else:
                        result[current_section] += "\n" + clean_line.capitalize()
                elif line:
                    result[current_section] += " " + line.capitalize()

    # Clean up results
    result["suggested_topics"] = result["suggested_topics"][:5]  # Limit to 5 topics
    for key in result:
        if isinstance(result[key], str):
            result[key] = result[key].replace("Not specified", "").strip() or "Not specified"
    
    return result

def extract_data_from_text(llm, file_path, knowledge_base=None):
    """Extract structured marketing data using text-based parsing"""
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