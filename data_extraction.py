# data_extraction.py
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredFileLoader

# Define prompt template for marketing data extraction
qa_prompt = PromptTemplate(
    template="""
    Based on the document content below, extract the following marketing data:

    {context}

    Extract:
    - Brand description
    - Target audience
    - Products or services
    - Marketing goals
    - Existing content
    - Keywords
    - Suggested topics for social media (3-5 topics)

    Use "Not specified" if information is missing. Provide the output in JSON format like this:
    {{
        "brand_description": "...",
        "target_audience": "...",
        "products_services": "...",
        "marketing_goals": "...",
        "existing_content": "...",
        "keywords": "...",
        "suggested_topics": ["...", "...", "..."]
    }}

    """,
    input_variables=["context"]
)

def extract_data_from_text(llm, file_path, knowledge_base=None):
    """Extract structured marketing data using RAG if enabled, otherwise use direct LLM invocation."""
    if knowledge_base:
        # Use RAG with RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=knowledge_base,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        response = qa_chain.invoke({"query": "marketing data extraction"})
        try:
            return json.loads(response["result"])
        except json.JSONDecodeError:
            st.error("Failed to parse RAG response as JSON.")
            return {
                "brand_description": "Not specified",
                "target_audience": "Not specified",
                "products_services": "Not specified",
                "marketing_goals": "Not specified",
                "existing_content": "Not specified",
                "keywords": "Not specified",
                "suggested_topics": []
            }
    else:
        # Direct LLM invocation (fallback)
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        prompt = qa_prompt.format(context=text)
        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            return json.loads(response.content)
        except json.JSONDecodeError:
            st.error("Failed to parse direct LLM response as JSON.")
            return {
                "brand_description": "Not specified",
                "target_audience": "Not specified",
                "products_services": "Not specified",
                "marketing_goals": "Not specified",
                "existing_content": "Not specified",
                "keywords": "Not specified",
                "suggested_topics": []
            }