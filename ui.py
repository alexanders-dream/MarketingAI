# ui.py
import streamlit as st
from config import get_api_key
from file_handling import extract_text_from_file
from data_extraction import extract_data_from_text
from marketing_functions import generate_strategy, generate_campaign, generate_content, optimize_seo
from utils import fetch_models, ProviderHandler
from file_handling import validate_file
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

st.set_page_config(
        page_title="Marketing Agent Pro",
        page_icon="📈",
        layout="wide"
    )
    
# Custom CSS injection
st.markdown("""
<style>
    .stTextInput label, .stTextArea label, .stSelectbox label { 
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    .stAlert { 
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .card {
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        background: white;
    }
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
        task = st.selectbox(
            "🎯 Select Task",
            ["Marketing Strategy", "Campaign Ideas", "Social Media Content", "SEO Optimization"],
            key="task_select"
        )
        
    return task

def initialize_llm():
    """Initialize the LLM client with current settings"""
    provider = st.session_state.get("provider_select")
    return ProviderHandler.create_client(
        provider=provider,
        model=st.session_state.get("model"),
        api_key=st.session_state.get("api_key"),
        endpoint=st.session_state.get("endpoint")
    )

def render_file_upload():
    """File upload section with validation"""
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
        
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.status("🔍 Analyzing document...", expanded=True) as status:
                try:
                    st.write("Extracting text...")
                    text = extract_text_from_file(uploaded_file)
                    
                    st.write("Identifying key elements...")
                    llm = initialize_llm()
                    extracted_data = extract_data_from_text(llm, text)
                    
                    st.session_state.extracted_data = extracted_data
                    st.session_state.current_file = uploaded_file.name
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="Analysis failed", state="error")
                    st.error(f"Error processing file: {str(e)}")
                    return None
        else:
            st.success("✅ Using previously analyzed document")
            
    return uploaded_file

def render_task_interface(llm, task):
    """Render the appropriate task interface"""
    with st.container():
        st.header(f"📋 {task}")
        
        if task == "Marketing Strategy":
            with st.form("strategy_form"):
                brand = st.text_area("Brand Description", 
                                   value=st.session_state.extracted_data.get("brand_description", ""))
                audience = st.text_area("Target Audience", 
                                      value=st.session_state.extracted_data.get("target_audience", ""))
                
                if st.form_submit_button("🚀 Generate Strategy"):
                    with st.spinner("Crafting strategy..."):
                        result = generate_strategy(llm, brand, audience)
                        st.session_state.result = result
            
        elif task == "Campaign Ideas":
            with st.form("campaign_form"):
                product_service = st.text_area("Products/Services", 
                                               value=st.session_state.extracted_data.get("products_services", ""))
                goals = st.text_area("Marketing Goals", 
                                         value=st.session_state.extracted_data.get("marketing_goals", ""))
                
                if st.form_submit_button("🚀 Generate Campaign"):
                    with st.spinner("Crafting campaign..."):
                        result = generate_campaign(llm, product_service, goals)
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
                        result = generate_content(llm, platform, topic, tone, target_audience)
                        st.session_state.result = result
        
        elif task == "SEO Optimization":
            with st.form("seo_form"):
                #content_source = st.radio("Content Source", ["Extracted", "Custom"])
                content = st.text_area("Content", value=st.session_state.extracted_data.get("existing_content", ""))
                keywords = st.text_input("Keywords (comma-separated)", value=st.session_state.extracted_data.get("keywords", ""))
                
                if st.form_submit_button("🚀 Generate Content"):
                    with st.spinner("Crafting content..."):
                        result = optimize_seo(llm, content, keywords)
                        st.session_state.result = result
            pass
        
        if 'result' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state.result)
                st.download_button("💾 Download", st.session_state.result, f"{task.replace(' ', '_')}.md")

def main():
    
    # Main header
    st.title("Marketing Agent Pro")
    st.markdown("---")
    
    # Setup sidebar and get selected task
    task = setup_sidebar()
    
    # Initialize LLM client
    llm = initialize_llm()
    
    # Main content area
    with st.container():
        tab_analysis, tab_manual = st.tabs(["📄 Document Analysis", "✍️ Manual Input"])
        
        with tab_analysis:
            uploaded_file = render_file_upload()
        
        with tab_manual:
            st.info("Coming soon: Direct input without document upload")
        
        if 'extracted_data' in st.session_state:
            render_task_interface(llm, task)
        else:
            st.info("✨ Upload a document or use manual input to get started")

if __name__ == "__main__":
    main()