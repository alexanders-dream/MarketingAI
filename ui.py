# ui.py
import streamlit as st
from config import get_api_key  # Loads API keys from .env
from file_handling import extract_text_from_file
from data_extraction import extract_data_from_text
from marketing_functions import generate_strategy, generate_campaign, generate_content, optimize_seo
from utils import fetch_models
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

st.set_page_config(
    page_title="Market Agent",
    #page_icon="images/icon.png"
)

def setup_sidebar():
    # Sidebar title
    st.sidebar.title("Marketing Agent")
    
    # Task selection section
    st.sidebar.subheader("Select Task")
    task = st.sidebar.selectbox(
        "Choose a task",
        ["Marketing Strategy", "Campaign Ideas", "Social Media Content", "SEO Optimization"],
        key="task_select"
    )
    
    # AI configuration section
    st.sidebar.subheader("AI Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Select AI Provider",
        ["Groq", "OpenAI", "Ollama"],
        key="provider_select"
    )
    
    # API Endpoint (editable, with defaults per provider)
    if "endpoints" not in st.session_state:
        st.session_state.endpoints = {}
    
    default_endpoints = {
        "Groq": "https://api.groq.com/openai/v1",
        "OpenAI": "https://api.openai.com/v1",
        "Ollama": "http://localhost:11434"
    }
    
    current_endpoint = st.session_state.endpoints.get(provider, default_endpoints[provider])
    endpoint = st.sidebar.text_input(
        "API Endpoint",
        value=current_endpoint,
        key=f"endpoint_input_{provider}"
    )
    st.session_state.endpoints[provider] = endpoint
    
    # API Key (only for providers that require it, prefilled from .env, editable, with show/hide toggle)
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {}
    
    if provider != "Ollama":  # Only show API key input for Groq and OpenAI
        default_api_key = get_api_key(provider)  # Expects GROQ_API_KEY, OPENAI_API_KEY, etc., in .env
        current_api_key = st.session_state.api_keys.get(provider, default_api_key or "")
        
        if f"show_api_key_{provider}" not in st.session_state:
            st.session_state[f"show_api_key_{provider}"] = False
        
        
        
        if st.session_state[f"show_api_key_{provider}"]:
            api_key = st.sidebar.text_input(
                f"{provider} API Key",
                value=current_api_key,
                key=f"api_key_input_{provider}_text"
            )
        else:
            api_key = st.sidebar.text_input(
                f"{provider} API Key",
                value=current_api_key,
                type="password",
                key=f"api_key_input_{provider}_password"
            )
        st.session_state.api_keys[provider] = api_key
    else:
        st.session_state.api_keys[provider] = None  # No API key needed for Ollama
    
    # Link to get API key (provider-specific)
    if provider == "Groq":
        st.sidebar.markdown('[Get Groq API Key](https://console.groq.com/keys)', unsafe_allow_html=True)
    elif provider == "OpenAI":
        st.sidebar.markdown('[Get OpenAI API Key](https://openai.com/api)', unsafe_allow_html=True)
    # Add Ollama link if applicable
    
    # Model selection (fetch models based on provider, endpoint, and API key)
    selected_model = None
    if provider == "Ollama" or st.session_state.api_keys.get(provider):  # Fetch models for Ollama without API key
        with st.spinner("Fetching models..."):
            api_key = st.session_state.api_keys.get(provider) if provider != "Ollama" else None
            models = fetch_models(provider, st.session_state.endpoints[provider], api_key)
        if models:
            selected_model = st.sidebar.selectbox("Select AI Model", models, key="model_select")
        else:
            st.sidebar.warning("No models available or failed to fetch models.")
    else:
        st.sidebar.warning("Please enter the API key to fetch models.")
    
    st.info(f"Using: {provider} - {selected_model if selected_model else 'No model selected'}")
    
    # Refresh models button
    if st.sidebar.button("Refresh Models"):
        st.cache_data.clear()  # Clears cache to refetch models
    
    # End session button
    if st.sidebar.button("End Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()  # Resets the app
    
    return task

# Function to instantiate LLM based on provider
def get_llm(provider, model, api_key, endpoint):
    if provider == "Groq":
        return ChatGroq(
            model=model, 
            api_key=api_key, 
            base_url="https://api.groq.com/", 
            temperature=0.7, 
            max_tokens=1000)
    
    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model, 
            api_key=api_key, 
            base_url=endpoint, 
            temperature=0.7, 
            max_tokens=1000)
    
    elif provider == "Ollama":
        return ChatOllama(
            model=model,
            base_url=endpoint,
            temperature=0.7
        )  # No API key needed for Ollama
    return None

def main():
    # Main title updated to match sidebar
    st.title("Marketing Agent")
    task = setup_sidebar()
    
    # Dynamic LLM creation based on sidebar inputs
    provider = st.session_state.get("provider_select")
    if provider:
        api_key = st.session_state.api_keys.get(provider)
        endpoint = st.session_state.endpoints.get(provider)
        model = st.session_state.get("model_select")
        if (provider == "Ollama" or api_key) and model:  # Allow Ollama without API key
            llm = get_llm(provider, model, api_key, endpoint)
            if llm:
                # File upload and task handling
                uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", ["pdf", "docx", "txt"])
                
                if uploaded_file is not None:
                    current_file_id = uploaded_file.name + str(uploaded_file.size)
                    
                    if 'previous_file_id' not in st.session_state or st.session_state.previous_file_id != current_file_id:
                        with st.spinner("Extracting text..."):
                            text = extract_text_from_file(uploaded_file)
                        if text:
                            with st.spinner("Extracting data..."):
                                extracted_data = extract_data_from_text(llm, text)
                                if extracted_data:
                                    st.session_state.extracted_data = extracted_data
                                    st.session_state.previous_file_id = current_file_id
                                    st.success("Data extracted successfully!")
                    else:
                        st.write("Using previously extracted data.")
                    
                    # Re-extract data option
                    if 'extracted_data' in st.session_state:
                        if st.button("Re-extract Data"):
                            with st.spinner("Extracting text..."):
                                text = extract_text_from_file(uploaded_file)
                            with st.spinner("Extracting data..."):
                                extracted_data = extract_data_from_text(llm, text)
                            st.session_state.extracted_data = extracted_data
                            st.success("Data re-extracted successfully!")
                else:
                    if 'extracted_data' in st.session_state:
                        del st.session_state.extracted_data
                    if 'previous_file_id' in st.session_state:
                        del st.session_state.previous_file_id
                
                # Task execution
                if task == "Marketing Strategy":
                    st.header("Generate Marketing Strategy")
                    default_brand = st.session_state.extracted_data.get("brand_description", "") if 'extracted_data' in st.session_state else ""
                    default_audience = st.session_state.extracted_data.get("target_audience", "") if 'extracted_data' in st.session_state else ""
                    brand_description = st.text_area("Brand Description", default_brand)
                    target_audience = st.text_area("Target Audience", default_audience)
                    if st.button("Generate Strategy"):
                        if brand_description and target_audience:
                            with st.spinner("Generating..."):
                                strategy = generate_strategy(llm, brand_description, target_audience)
                            st.write(strategy)
                            st.download_button("Download", strategy, "strategy.txt")
                        else:
                            st.warning("Please provide brand description and target audience.")
                
                elif task == "Campaign Ideas":
                    st.header("Generate Campaign Ideas")
                    default_products = st.session_state.extracted_data.get("products_services", "") if 'extracted_data' in st.session_state else ""
                    default_goals = st.session_state.extracted_data.get("marketing_goals", "") if 'extracted_data' in st.session_state else ""
                    product_service = st.text_area("Products/Services", default_products)
                    goals = st.text_area("Marketing Goals", default_goals)
                    if st.button("Generate Ideas"):
                        if product_service and goals:
                            with st.spinner("Generating..."):
                                ideas = generate_campaign(llm, product_service, goals)
                            st.write(ideas)
                            st.download_button("Download", ideas, "campaign_ideas.txt")
                        else:
                            st.warning("Please provide products/services and goals.")
                
                elif task == "Social Media Content":
                    st.header("Generate Social Media Content")
                    suggested_topics = st.session_state.extracted_data.get("suggested_topics", []) if 'extracted_data' in st.session_state else []
                    default_audience = st.session_state.extracted_data.get("target_audience", "") if 'extracted_data' in st.session_state else ""
                    topic_options = ["Custom topic"] + suggested_topics
                    selected_topic = st.selectbox("Topic", topic_options)
                    topic = st.text_input("Custom Topic", "") if selected_topic == "Custom topic" else selected_topic
                    platform = st.selectbox("Platform", ["Instagram", "LinkedIn", "TikTok", "Facebook"])
                    tone = st.selectbox("Tone", ["Formal", "Casual", "Humorous", "Inspirational"])
                    target_audience = st.text_area("Target Audience", default_audience)
                    if st.button("Generate Content"):
                        if topic:
                            with st.spinner("Generating..."):
                                content = generate_content(llm, platform, topic, tone, target_audience)
                            st.write(content)
                            st.download_button("Download", content, "post.txt")
                        else:
                            st.warning("Please provide a topic.")
                
                elif task == "SEO Optimization":
                    st.header("Optimize SEO")
                    extracted_content = st.session_state.extracted_data.get("existing_content", "") if 'extracted_data' in st.session_state else ""
                    default_keywords = st.session_state.extracted_data.get("keywords", "") if 'extracted_data' in st.session_state else ""
                    content_source = st.radio("Content Source", ["Extracted", "Custom"])
                    content = st.text_area("Content", extracted_content if content_source == "Extracted" else "")
                    keywords = st.text_input("Keywords (comma-separated)", default_keywords)
                    if st.button("Optimize SEO"):
                        if content and keywords:
                            with st.spinner("Generating..."):
                                suggestions = optimize_seo(llm, content, keywords)
                            st.write(suggestions)
                            st.download_button("Download", suggestions, "seo_suggestions.txt")
                        else:
                            st.warning("Please provide content and keywords.")
            else:
                st.warning("Failed to create LLM instance.")
        else:
            st.warning("Please select a model and enter the API key (if required).")
    else:
        st.warning("Please select a provider.")

if __name__ == "__main__":
    main()