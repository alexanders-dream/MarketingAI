import streamlit as st
import os
from config import get_api_key, config
from file_handling import extract_text_from_file
from data_extraction import extract_data_from_text
from marketing_functions import generate_strategy, generate_campaign, generate_content, optimize_seo
from utils import fetch_models, ProviderHandler
from file_handling import validate_file
from rag_utils import get_knowledge_base, working_dir
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

"""st.set_page_config(
        page_title="Marketing Agent Pro",
        page_icon="📈",
        #layout="wide"
    )"""
  
# Custom CSS injection
st.markdown("""
<style>
    .stTextInput label, .stTextArea label, .stSelectbox label { 
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    .stAlert { 
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        background-color: #2c3e50;
    }
    .card {
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        background: grey;
    }
    /* Enhanced card styling */
    .stContainer {
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Better form spacing */
    .stForm > div {
        gap: 1.2rem !important;
    }
    
    /* Progress indicators */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .pulse { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)

def setup_sidebar():
    """Configure the sidebar with AI settings"""
    st.sidebar.title("⚙️ AI Configuration")
    
    with st.sidebar:
        provider = st.selectbox(
            "AI Provider",
            ["Groq", "OpenAI", "Ollama"],
            key="provider_select",
            help="Select your preferred AI service provider"
        )

        with st.expander("Model Settings", expanded=False):
            st.session_state.temperature = st.slider(
                "Creativity (Temperature)",
                0.0, 1.0, config.TEMPERATURE,
                help="Higher values = more creative, lower = more focused"
            )
        
        with st.expander("Advanced Settings", expanded=False):
            default_endpoints = {
                "Groq": "https://api.groq.com/openai/v1",
                "OpenAI": "https://api.openai.com/v1",
                "Ollama": "http://localhost:11434"
            }
            
            endpoint = st.text_input(
                "API Endpoint",
                value=st.session_state.get("endpoint", default_endpoints[provider]),
                key="endpoint_input"
            )
            st.session_state.endpoint = endpoint
            
            if provider != "Ollama":
                api_key = st.text_input(
                    f"{provider} API Key",
                    type="password",
                    value=get_api_key(provider),
                    help=f"Get your API key from {provider}'s dashboard"
                )
                st.session_state.api_key = api_key
            else:
                st.session_state.api_key = None
                

        # Model selection with caching
        if provider == "Ollama" or st.session_state.get("api_key"):
            with st.spinner("Loading models..."):
                models = fetch_models(
                    provider,
                    endpoint,
                    st.session_state.get("api_key")
                )
                
            model = st.selectbox(
                "AI Model",
                models,
                key="model_select",
                help="Select the model version to use"
            )
            st.session_state.model = model
            
        st.sidebar.markdown("---")
        st.sidebar.title("🎯 Select Task")
        task = st.selectbox(
            "🎯 Select Task",
            ["Marketing Strategy", "Campaign Ideas", "Social Media Content", "SEO Optimization"],
            key="task_select"
        )

        use_rag = st.sidebar.checkbox("Use RAG (Enhanced Extraction)", value=True)
       
    return task, use_rag

def initialize_llm():
    """Initialize the LLM client with current settings"""
    
    provider = st.session_state.get("provider_select")
    if st.session_state.get("api_key") == "":
        st.info("Add a valid API key. Edit Advanced Settings")
    else:
        return ProviderHandler.create_client(
            provider=provider,
            model=st.session_state.get("model"),
            api_key=st.session_state.get("api_key"),
            endpoint=st.session_state.get("endpoint")
        )

def render_file_upload(use_rag):
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Marketing Document",
            type=["pdf", "docx", "txt"],
            help="Max file size: 5MB"
        )
    
    if uploaded_file:
        if not validate_file(uploaded_file):
            st.error("Invalid file. Please check size and format.")
            return None
        
        # Save uploaded file temporarily
        file_id = uploaded_file.name + str(uploaded_file.size)
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if 'current_file_id' not in st.session_state or st.session_state.current_file_id != file_id:
            with st.status("🔍 Analyzing document...", expanded=True) as status:
                try:
                    knowledge_base = None
                    if use_rag:
                        st.write("Building knowledge base for RAG...")
                        knowledge_base = get_knowledge_base(file_id, file_path)
                    
                    st.write("Extracting marketing data...")
                    llm = initialize_llm()  # Your existing LLM initialization
                    extracted_data = extract_data_from_text(llm, file_path, knowledge_base)
                    
                     # Initialize session state with proper structure
                    st.session_state.extracted_data = {
                        "brand_description": extracted_data.get("brand_description", ""),
                        "target_audience": extracted_data.get("target_audience", ""),
                        "products_services": extracted_data.get("products_services", ""),
                        "marketing_goals": extracted_data.get("marketing_goals", ""),
                        "existing_content": extracted_data.get("existing_content", ""),
                        "keywords": extracted_data.get("keywords", ""),
                        "suggested_topics": extracted_data.get("suggested_topics", [])
                    }
                    
                    # Update UI fields directly
                    for key in st.session_state.extracted_data:
                        if key in st.session_state:
                            st.session_state[key] = extracted_data.get(key, "")
                    
                except Exception as e:
                    status.update(label="Analysis failed", state="error")
                    st.error(f"Error processing file: {str(e)}")
                    return None
                finally:
                    # Clean up temporary file
                    if os.path.exists(file_path):
                        os.remove(file_path)
        else:
            st.success("✅ Using previously analyzed document")
            
    return uploaded_file

def render_task_interface(llm, task):
    """Render the appropriate task interface"""
    temperature = st.session_state.get("temperature", config.TEMPERATURE)

    if task not in ["Marketing Strategy", "Campaign Ideas", "Social Media Content", "SEO Optimization"]:
        st.error("Invalid task selection")
        return
    
    with st.container():
        st.header(f"📋 {task}")
        
        if task == "Marketing Strategy":
            with st.form("strategy_form"):
                brand = st.text_area("Brand Description", 
                                   value=st.session_state.extracted_data.get("brand_description", ""))
                audience = st.text_area("Target Audience", 
                                      value=st.session_state.extracted_data.get("target_audience", ""))
                
                if st.form_submit_button("🚀 Generate Strategy"):
                    with st.spinner("Crafting Strategy..."):
                        result = generate_strategy(llm, brand, audience, temperature)
                        st.session_state.result = result
            
        elif task == "Campaign Ideas":
            with st.form("campaign_form"):
                product_service = st.text_area("Products/Services", 
                                               value=st.session_state.extracted_data.get("products_services", ""))
                goals = st.text_area("Marketing Goals", 
                                         value=st.session_state.extracted_data.get("marketing_goals", ""))
                
                if st.form_submit_button("🚀 Generate Campaign"):
                    with st.spinner("Crafting Campaign..."):
                        result = generate_campaign(llm, product_service, goals, temperature)
                        st.session_state.result = result

        elif task == "Social Media Content":
            with st.form("social_form"):
                suggested_topics = st.session_state.extracted_data.get("suggested_topics", []) if 'extracted_data' in st.session_state else []
                topic_options = ["Custom topic"] + suggested_topics
                selected_topic = st.selectbox("Topic", topic_options)
                topic = st.text_input("Custom Topic", "") if selected_topic == "Custom topic" else selected_topic
                platform = st.selectbox("Platform", ["Instagram", "LinkedIn", "TikTok", "Facebook"])
                tone = st.selectbox("Tone", ["Formal", "Casual", "Humorous", "Inspirational"])
                target_audience = st.text_area("Target Audience", value=st.session_state.extracted_data.get("target_audience", ""))
                
                if st.form_submit_button("🚀 Generate Content"):
                    with st.spinner("Crafting content..."):
                        result = generate_content(llm, platform, topic, tone, target_audience, temperature)
                        st.session_state.result = result
        
        elif task == "SEO Optimization":
            with st.form("seo_form"):
                #content_source = st.radio("Content Source", ["Extracted", "Custom"])
                content = st.text_area("Content", value=st.session_state.extracted_data.get("existing_content", ""))
                keywords = st.text_input("Keywords (comma-separated)", value=st.session_state.extracted_data.get("keywords", ""))
                
                if st.form_submit_button("🚀 Generate SEO strategy"):
                    with st.spinner("Crafting SEO strategy..."):
                        result = optimize_seo(llm, content, keywords, temperature)
                        st.session_state.result = result
            pass
        
        if 'result' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state.result)
                st.download_button("💾 Download", st.session_state.result, f"{task.replace(' ', '_')}.md")

