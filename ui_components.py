"""
Reusable UI components for Marketing AI v3
"""
import streamlit as st
import pandas as pd
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from io import BytesIO
from datetime import datetime

from config import AppConfig, DatabaseConfig
from llm_handler import LLMProviderHandler
from database import DatabaseManager
from document_processor import DocumentProcessor
from web_scraper import WebScraper
from prompts import Prompts
import re


def extract_json_from_text(text: Any) -> Dict[str, Any]:
    """
    Extract JSON object from LLM response text using robust parsing with fallbacks.
    Handles cases where JSON is embedded in other text or has formatting issues.
    Also maps field names to ensure consistency.
    """
    # Handle various input types gracefully
    if not text:
        return {}
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return {}
    
    # Clean the text first - remove markdown code blocks and extra whitespace
    cleaned_text = text.strip()
    
    # Remove markdown code blocks if present
    if cleaned_text.startswith('```'):
        lines = cleaned_text.split('\n')
        if len(lines) > 2 and lines[0].startswith('```') and lines[-1].startswith('```'):
            cleaned_text = '\n'.join(lines[1:-1]).strip()
    
    # Try direct parsing first with comprehensive error handling
    try:
        parsed_data = json.loads(cleaned_text)
        return map_field_names(parsed_data)
    except json.JSONDecodeError as e:
        # Log the error but continue with fallback methods
        pass
    
    # Enhanced JSON extraction with multiple fallback strategies
    extraction_attempts = [
        _extract_json_using_brackets,
        _extract_json_using_regex,
        _extract_json_using_llm_fallback
    ]
    
    for extraction_method in extraction_attempts:
        try:
            result = extraction_method(cleaned_text)
            if result:
                return map_field_names(result)
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Final fallback: try to extract any JSON-like structure
    try:
        # Look for the first { and last } in the text with better error handling
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned_text[start_idx:end_idx+1]
            # Try to fix common JSON issues
            json_str = _fix_common_json_issues(json_str)
            parsed_data = json.loads(json_str)
            return map_field_names(parsed_data)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # If no valid JSON found, provide helpful error message
    st.error("❌ Could not parse AI response as valid JSON. The AI may have returned malformed output.")
    st.info("💡 Try rephrasing your prompt or asking the AI to format the response as pure JSON.")
    return {}


def _extract_json_using_brackets(text: str) -> Dict[str, Any]:
    """Extract JSON using bracket matching with validation"""
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    # Found complete JSON object
                    json_str = text[start_idx:i+1]
                    return json.loads(json_str)
    
    raise ValueError("No complete JSON object found")


def _extract_json_using_regex(text: str) -> Dict[str, Any]:
    """Extract JSON using regex patterns with validation"""
    # More robust regex pattern for JSON objects
    json_pattern = r'\{[\s\S]*?\}(?=\s*\{|$)'  # Match complete JSON objects
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Clean and validate the match
            match = match.strip()
            if match.startswith('{') and match.endswith('}'):
                # Try to fix common issues before parsing
                match = _fix_common_json_issues(match)
                return json.loads(match)
        except (json.JSONDecodeError, ValueError):
            continue
    
    raise ValueError("No valid JSON found via regex")


def _extract_json_using_llm_fallback(text: str) -> Dict[str, Any]:
    """
    Fallback method that uses simple string processing when JSON parsing fails.
    This is a last resort for badly formatted but still parseable content.
    """
    # Look for key-value patterns in the text
    lines = text.split('\n')
    result = {}
    
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().strip('"\'')
                value = parts[1].strip().strip('"\'')
                if key and value:
                    result[key] = value
    
    if result:
        return result
    else:
        raise ValueError("No parseable key-value pairs found in text")


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues"""
    if not json_str:
        return json_str
    
    # Fix missing quotes around keys
    json_str = re.sub(r'(\w+)(\s*:\s*)', r'"\1"\2', json_str)
    
    # Fix single quotes to double quotes
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Remove comments (JSON doesn't support comments)
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    return json_str


def map_field_names(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map various field name variations to the expected field names.
    """
    if not isinstance(data, dict):
        return {}

    # Define field name mappings (common variations to expected names)
    field_mappings = {
        # Company name variations
        "company": "company_name",
        "company_name": "company_name",
        "organization": "company_name",
        "business_name": "company_name",

        # Industry variations
        "industry": "industry",
        "sector": "industry",
        "field": "industry",

        # Target audience variations
        "target_audience": "target_audience",
        "audience": "target_audience",
        "customers": "target_audience",
        "target_market": "target_audience",

        # Products/Services variations
        "products_services": "products_services",
        "products": "products_services",
        "services": "products_services",
        "offerings": "products_services",

        # Brand description variations
        "brand_description": "brand_description",
        "brand": "brand_description",
        "description": "brand_description",
        "about": "brand_description",
        "mission": "brand_description",

        # Marketing goals variations
        "marketing_goals": "marketing_goals",
        "goals": "marketing_goals",
        "objectives": "marketing_goals",
        "marketing_objectives": "marketing_goals",

        # Existing content variations
        "existing_content": "existing_content",
        "content": "existing_content",
        "current_content": "existing_content",
        "marketing_content": "existing_content",

        # Keywords variations
        "keywords": "keywords",
        "keyword": "keywords",
        "seo_keywords": "keywords",

        # Market opportunities variations
        "market_opportunities": "market_opportunities",
        "opportunities": "market_opportunities",
        "growth_areas": "market_opportunities",

        # Competitive advantages variations
        "competitive_advantages": "competitive_advantages",
        "advantages": "competitive_advantages",
        "competitors": "competitive_advantages",
        "differentiators": "competitive_advantages",

        # Customer pain points variations
        "customer_pain_points": "customer_pain_points",
        "pain_points": "customer_pain_points",
        "problems": "customer_pain_points",
        "challenges": "customer_pain_points",
    }

    # Create mapped result
    mapped_result = {}

    for key, value in data.items():
        # Map the key if it exists in mappings, otherwise keep original
        mapped_key = field_mappings.get(key.lower(), key)
        mapped_result[mapped_key] = value

    return mapped_result


def convert_to_docx(content: str) -> bytes:
    """Convert markdown content to DOCX format"""
    document = Document()
    content = content.strip()

    # Split content into paragraphs
    paragraphs = content.split('\n\n')

    for paragraph in paragraphs:
        if paragraph.startswith('# '):
            # Heading
            heading = document.add_heading(paragraph[2:], level=1)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif paragraph.startswith('## '):
            # Subheading
            heading = document.add_heading(paragraph[3:], level=2)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif paragraph.startswith('- '):
            # Bullet point
            document.add_paragraph().add_run(paragraph).font.size = Pt(10)
        else:
            # Regular paragraph
            document.add_paragraph(paragraph)

    # Save to bytes
    file_bytes = BytesIO()
    document.save(file_bytes)
    file_bytes.seek(0)
    return file_bytes.read()


class SidebarManager:
    """Manages the application sidebar"""

    def __init__(self):
        self.llm_manager = None

    def create_sidebar(self) -> Dict[str, Any]:
        """Create the sidebar UI components with smooth model switching"""
        default_endpoints = AppConfig.PROVIDER_ENDPOINTS

        with st.sidebar:
            self._create_upgrade_section()

            st.header("🎯 Marketing Task")
            task = st.selectbox(
                "Select Task",
                options=AppConfig.MARKETING_TASKS,
                key="task_select",
                on_change=self._on_task_change
            )

            st.title("⚙️ AI Configuration")

            # Provider selection with callback
            provider = st.selectbox(
                "AI Provider",
                options=["Groq", "OpenAI", "Gemini", "Ollama"],
                key="provider_select",
                help="Select your preferred AI service provider",
                on_change=self._on_provider_change
            )

            with st.expander("Provider Settings", expanded=True):
                # API Configuration
                if provider == "Groq":
                    default_endpoint = default_endpoints["GROQ"]
                elif provider == "OpenAI":
                    default_endpoint = default_endpoints["OPENAI"]
                elif provider == "Gemini":
                    default_endpoint = default_endpoints["GEMINI"]
                else:  # Ollama
                    default_endpoint = default_endpoints["OLLAMA"]

                endpoint = st.text_input(
                    "API Endpoint",
                    value=default_endpoint,
                    key="endpoint_input",
                    on_change=self._on_config_change
                )

                api_key = None

                if provider != "Ollama":
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        api_key = st.text_input(
                            f"{provider} API Key",
                            type="password",
                            value=AppConfig.get_api_key(provider),
                            help=f"Get your API key from {provider}'s dashboard",
                            key="api_key_input",
                            on_change=self._on_config_change
                        )
                    with col2:
                        st.button("Validate", key=f"validate_{provider}", on_click=self._validate_api_key, args=(provider, endpoint, api_key))

                    if provider == "Groq":
                        st.sidebar.markdown("[Get Groq API Key](https://console.groq.com/keys)")
                    elif provider == "OpenAI":
                        st.sidebar.markdown("[Get OpenAI API Key](https://platform.openai.com/api-keys)")
                    elif provider == "Gemini":
                        st.sidebar.markdown("[Get Gemini API Key](https://makersuite.google.com/app/apikey)")
                else:
                    api_key = None
                    st.sidebar.markdown("[Download Ollama](https://ollama.com/)")

                # Model selection with smooth switching
                model = self._create_model_selector(provider, endpoint, api_key)

            with st.expander("Advanced Settings", expanded=False):
                temperature = st.slider(
                    "Temperature", 0.0, 1.0, AppConfig.DEFAULT_TEMPERATURE,
                    help="High temp = high creativity",
                    key="temperature_slider",
                    on_change=self._on_config_change
                )
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=512,
                    max_value=8192,
                    value=AppConfig.DEFAULT_MAX_TOKENS,
                    key="max_tokens_input",
                    on_change=self._on_config_change
                )

        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "api_endpoint": endpoint,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "task": task,
        }

    def _create_model_selector(self, provider: str, endpoint: str, api_key: str = None) -> str:
        """Create model selector with smooth switching"""
        model = None

        if provider == "Ollama" or api_key:
            # Cache models to avoid repeated API calls
            cache_key = f"models_{provider}_{endpoint}"
            if cache_key not in st.session_state or st.button("🔄 Refresh Models", key="refresh_models"):
                with st.spinner("Loading models..."):
                    models, error = LLMProviderHandler.fetch_models(provider, endpoint, api_key)
                    if error:
                        st.error(f"Failed to load models: {error}")
                        st.session_state[cache_key] = ([], error)
                    else:
                        st.session_state[cache_key] = (models, None)

            models, error = st.session_state.get(cache_key, ([], None))

            if models:
                # Get current model from session state, default to first model
                current_model = st.session_state.get("model", models[0] if models else None)

                model = st.selectbox(
                    "Select AI Model",
                    models,
                    index=models.index(current_model) if current_model in models else 0,
                    key="model_select",
                    help="Select the model version to use",
                    on_change=self._on_model_change
                )
            elif not error:
                st.warning(f"No models available for {provider}. Click '🔄 Refresh Models' to retry.")

        return model

    def _validate_api_key(self, provider: str, endpoint: str, api_key: str):
        """Validate the API key by fetching models"""
        _, error = LLMProviderHandler.fetch_models(provider, endpoint, api_key)
        if error:
            st.error(f"API key validation failed: {error}")
        else:
            st.success("API key is valid!")

    def _on_provider_change(self):
        """Handle provider change - clear model cache"""
        # Clear model cache when provider changes
        st.session_state.model = None
        # Clear any cached models for different providers
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("models_")]
        for key in keys_to_remove:
            del st.session_state[key]

    def _on_task_change(self):
        """Handle task change"""
        st.session_state.task = st.session_state.task_select

    def _on_model_change(self):
        """Handle model change - prepare for smooth switching"""
        # Mark that model changed for smooth transition
        st.session_state.model_changed = True

    def _on_config_change(self):
        """Handle configuration changes"""
        # Mark config as changed
        st.session_state.config_changed = True

    def _create_upgrade_section(self):
        """Create the upgrade/features section"""
        with st.expander("🚀 Unlock Extra Features", expanded=False):
            st.markdown(
                """
                <div style="margin-left: 0px; margin-top: 20px;">
                    <a href="https://calendly.com/alexanderoguso/30min" target="_blank">
                        <div style="background-color: #33353d; padding: 15px 30px; border-radius: 8px; text-align: center; width: 250px; height: 54px; display: flex; align-items: center; justify-content: center; transition: all 0.3s ease;">
                            <span style="color: white; font-weight: 600; font-size: 16px;">📞 Need more? Book a call</span>
                        </div>
                    </a>
                    <p style="color: #ffffff; font-weight: 600; margin-top: 5px; text-align: center;">Let's chat</p>
                </div>
                """, unsafe_allow_html=True
            )

            st.markdown(
                """<div style="text-align: center; margin-top: 20px;">
                    <a href="https://buymeacoffee.com/oguso">
                        <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="width: 150px; height: auto;">
                    </a>
                    <p style="color: #ffffff; margin-top: 5px;">Support my work!</p>
                </div>
                """, unsafe_allow_html=True
            )


class ProjectManager:
    """Manages project-related UI components"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_project_selector(self) -> Optional[int]:
        """Create project selection dropdown"""
        projects = self.db.get_projects()

        if not projects:
            st.info("No projects found. Create your first project below.")
            return None

        project_options = ["Select a project"] + [p["name"] for p in projects]
        selected_name = st.selectbox(
            "Select Project",
            options=project_options,
            help="Choose an existing project or create a new one"
        )

        if selected_name != "Select a project":
            selected_project = next(p for p in projects if p["name"] == selected_name)
            return selected_project["id"]

        return None

    def create_project_form(self) -> Optional[int]:
        """Create new project form"""
        with st.expander("Create New Project", expanded=False):
            with st.form("new_project_form"):
                col1, col2 = st.columns(2)

                with col1:
                    project_name = st.text_input("Project Name", placeholder="My Marketing Project")

                with col2:
                    project_description = st.text_area(
                        "Description (Optional)",
                        placeholder="Brief description of your project",
                        height=100
                    )

                if st.form_submit_button("Create Project"):
                    if project_name.strip():
                        project_id = self.db.create_project(
                            project_name.strip(),
                            project_description.strip()
                        )
                        st.success(f"Project '{project_name}' created successfully!")
                        st.rerun()
                        return project_id
                    else:
                        st.error("Project name is required.")

        return None

    def show_project_info(self, project_id: int):
        """Display project information"""
        project = self.db.get_project(project_id)
        if project:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Project", project["name"])

            with col2:
                content_count = len(self.db.get_project_content(project_id))
                st.metric("Content Pieces", content_count)

            with col3:
                st.metric("Created", project["created_at"][:10])


class DocumentUploader:
    """Handles document upload UI"""

    def create_uploader(self) -> Optional[Tuple[bytes, str]]:
        """Create file uploader component"""
        uploaded_file = st.file_uploader(
            "Upload business document (PDF, DOCX, TXT, MD)",
            type=AppConfig.SUPPORTED_FILE_TYPES,
            help="Upload documents to help the AI understand your business"
        )

        if uploaded_file:
            if self._validate_file(uploaded_file):
                return uploaded_file.getvalue(), uploaded_file.name
            else:
                st.error(f"Invalid file. Supported types: {', '.join(AppConfig.SUPPORTED_FILE_TYPES)}")
                return None

        return None

    def _validate_file(self, file) -> bool:
        """Validate uploaded file"""
        # Check file size
        if file.size > AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File size exceeds {AppConfig.MAX_FILE_SIZE_MB}MB limit")
            return False

        # Check file type
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in AppConfig.SUPPORTED_FILE_TYPES:
            return False

        return True


class ContentDisplay:
    """Handles content display and export"""

    @staticmethod
    def show_generated_content(content: str, task_type: str, project_id: Optional[int] = None):
        """Display generated content with export options"""
        st.subheader("Generated Content")
        st.markdown(content)

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            # Download as DOCX
            docx_file = convert_to_docx(content)
            st.download_button(
                label="📄 Download DOCX",
                data=docx_file,
                file_name=f"{task_type.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        with col2:
            # Copy to clipboard button
            if st.button("📋 Copy to Clipboard", help="Click to copy content to clipboard", key=f"copy_{task_type}"):
                try:
                    # Use JavaScript to copy to clipboard without causing rerun
                    import streamlit.components.v1 as components
                    escaped_content = content.replace('"', '\\"').replace("'", "\\'").replace("`", "\\`")
                    js_code = f"""
                    <script>
                    function copyToClipboard() {{
                        navigator.clipboard.writeText("{escaped_content}");
                        return false;
                    }}
                    copyToClipboard();
                    </script>
                    """
                    components.html(js_code, height=0)
                    st.success("✅ Content copied to clipboard!")
                except Exception as e:
                    st.error(f"Failed to copy: {str(e)}")

    @staticmethod
    def show_performance_score(score_data: Dict[str, Any]):
        """Display content performance scoring"""
        if "error" in score_data:
            st.error(f"Scoring failed: {score_data['error']}")
            return

        st.subheader("Content Performance Analysis")

        # Overall score
        overall_score = score_data.get("overall_score", 0)
        if overall_score >= 8:
            st.success(f"🎉 Excellent Performance (Score: {overall_score:.1f}/10)")
        elif overall_score >= 6:
            st.warning(f"👍 Good Performance (Score: {overall_score:.1f}/10)")
        else:
            st.error(f"⚠️ Needs Improvement (Score: {overall_score:.1f}/10)")

        # Individual scores
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Engagement", f"{score_data.get('engagement_score', 0)}/10")
            st.metric("SEO", f"{score_data.get('seo_score', 0)}/10")

        with col2:
            st.metric("Shareability", f"{score_data.get('shareability_score', 0)}/10")
            st.metric("Conversion", f"{score_data.get('conversion_score', 0)}/10")

        # Recommendations
        if "recommendations" in score_data and score_data["recommendations"]:
            with st.expander("Improvement Recommendations", expanded=True):
                for i, rec in enumerate(score_data["recommendations"], 1):
                    st.write(f"{i}. {rec}")


class BusinessContextManager:
    """Unified business context management with versioning and multi-source import"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.fields = DatabaseConfig.BUSINESS_CONTEXT_FIELDS
        self.doc_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.prompts = Prompts()
        
        # Initialize session state
        if "business_context" not in st.session_state:
            st.session_state.business_context = {}

    def display_context_manager(self, llm_client):
        """Main context manager interface"""
        st.header("🏢 Business Context Manager")
        
        # Load existing context if available
        project_id = st.session_state.get("current_project_id")
        if not project_id:
            st.info("👆 Please select or create a project first")
            return
            
        # Load latest context if not already loaded
        if not st.session_state.business_context:
            latest_context = self.db.get_latest_context(project_id)
            if latest_context:
                st.session_state.business_context = latest_context
                st.success("✅ Loaded existing business context")

        self._display_context_summary()
        self._display_import_sources(llm_client)
        self._display_version_management()
        self._display_validation_status()

    def _display_context_summary(self):
        """Display editable context summary"""
        st.subheader("📋 Business Information")
        
        # Ensure all fields exist in business_context
        for field_key, _, _ in self.fields:
            if field_key not in st.session_state.business_context:
                st.session_state.business_context[field_key] = ""
        
        # Create DataFrame for editing
        context_data = []
        for field_key, field_label, is_required in self.fields:
            value = st.session_state.business_context.get(field_key, "")
            # Handle different value types safely for status checking
            if isinstance(value, str):
                is_completed = bool(value.strip())
            elif isinstance(value, (list, dict)):
                is_completed = bool(value)  # Non-empty list or dict is considered completed
            else:
                is_completed = bool(value)  # Any other truthy value
                
            status = "✅" if is_completed else ("❗" if is_required else "⭕")
            context_data.append({
                "Field": field_label,
                "Status": status,
                "Value": value,
                "Required": "Yes" if is_required else "No"
            })
        
        df = pd.DataFrame(context_data)
        
        # Use a unique key that changes when business_context changes
        context_hash = hash(str(sorted(st.session_state.business_context.items())))
        
        # Display as editable table
        edited_df = st.data_editor(
            df,
            column_config={
                "Field": st.column_config.TextColumn("Field", disabled=True),
                "Status": st.column_config.TextColumn("Status", disabled=True, width="small"),
                "Value": st.column_config.TextColumn("Value", width="large"),
                "Required": st.column_config.TextColumn("Required", disabled=True, width="small")
            },
            hide_index=True,
            use_container_width=True,
            key=f"context_editor_{context_hash}"
        )
        
        # Update session state with edits
        for i, (field_key, _, _) in enumerate(self.fields):
            new_value = edited_df.iloc[i]["Value"]
            st.session_state.business_context[field_key] = new_value
        
        # Force immediate update of validation status
        self._update_validation_status()

    def _display_import_sources(self, llm_client):
        """Display import options from various sources"""
        st.subheader("📥 Import Business Information")
        
        tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "🌐 Website Analysis", "🤖 AI Enhancement"])
        
        with tab1:
            self._document_import_section(llm_client)
        
        with tab2:
            self._website_import_section(llm_client)
        
        with tab3:
            self._ai_enhancement_section(llm_client)

    def _document_import_section(self, llm_client):
        """Document analysis and import"""
        st.write("Upload business documents to extract context automatically")
        
        uploaded_file = st.file_uploader(
            "Choose document",
            type=AppConfig.SUPPORTED_FILE_TYPES,
            help="Upload business plans, marketing materials, or company documents"
        )
        
        if uploaded_file:
            if st.button("🔍 Analyze Document", type="primary"):
                with st.spinner("Analyzing document..."):
                    try:
                        _, text = self.doc_processor.load_document(uploaded_file.getvalue(), uploaded_file.name)
                        prompt = self.prompts.get_document_extraction_prompt(text)
                        from llm_handler import LLMManager
                        llm_manager = LLMManager()
                        insights_json = llm_manager.generate(llm_client, prompt)
                        insights = extract_json_from_text(insights_json)

                        # Debug: Test JSON parsing
                        st.write("🔍 **JSON Parsing Test:**")
                        st.write(f"Raw JSON length: {len(insights_json)}")
                        st.write(f"Parsed insights keys: {list(insights.keys()) if insights else 'None'}")
                        st.write(f"Company name from parsed: {insights.get('company_name', 'NOT FOUND') if insights else 'No insights'}")

                        if insights:
                            st.success("✅ Document analysis completed!")
                            
                            with st.expander("📋 Extracted Information", expanded=True):
                                st.json(insights)

                            # Automatically apply insights to the data table
                            st.info("💡 **Automatically applying extracted insights to your business context...**")
                            
                            # Apply the insights directly to session state
                            st.session_state.business_context.update(insights)

                            # Save to database
                            self._save_context_version("document")

                            # Force immediate validation update
                            self._update_validation_status()
                            
                            st.success("✅ Document insights automatically applied to your business context!")
                            st.balloons()

                            # Use Streamlit's built-in rerun mechanism with proper state management
                            st.session_state.document_analysis_completed = True
                            st.rerun()
                        else:
                            st.error("❌ Could not extract insights from document. Please try manual input.")
                    except Exception as e:
                        st.error(f"❌ Document analysis failed: {e}")

    def _website_import_section(self, llm_client):
        """Website analysis and import"""
        st.write("Analyze your website or social media profiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            website_url = st.text_input(
                "Website URL",
                placeholder="https://www.yourcompany.com"
            )
        
        with col2:
            social_url = st.text_input(
                "Social Media URL",
                placeholder="https://linkedin.com/company/yourcompany"
            )
        
        urls_to_analyze = []
        if website_url:
            urls_to_analyze.append(("website", website_url))
        if social_url:
            urls_to_analyze.append(("social", social_url))
        
        if urls_to_analyze and st.button("🌐 Analyze Online Presence", type="primary"):
            with st.spinner("Analyzing online presence..."):
                try:
                    all_insights = {}
                    for platform, url in urls_to_analyze:
                        content = self.web_scraper.scrape_website_content_sync(url)
                        prompt = self.prompts.get_website_extraction_prompt(content, platform)
                        from llm_handler import LLMManager
                        llm_manager = LLMManager()
                        insights_json = llm_manager.generate(llm_client, prompt)
                        insights = extract_json_from_text(insights_json)
                        all_insights.update(insights)

                    if all_insights:
                        st.success("✅ Online analysis completed!")
                        
                        with st.expander("🌐 Extracted Information", expanded=True):
                            st.json(all_insights)
                        
                        if st.button("✅ Apply Web Insights"):
                            st.session_state.business_context.update(all_insights)
                            self._save_context_version("web")
                            st.success("Web insights applied!")
                            st.rerun()
                    else:
                        st.error("❌ Could not analyze online presence. Please check URLs and try again.")
                except Exception as e:
                    st.error(f"❌ Web analysis failed: {e}")

    def _ai_enhancement_section(self, llm_client):
        """AI-powered context enhancement"""
        st.write("Let AI suggest improvements to your business information")
        
        current_context = st.session_state.business_context
        
        if st.button("🤖 Generate AI Suggestions", type="primary"):
            with st.spinner("Generating AI suggestions..."):
                try:
                    prompt = self.prompts.get_business_context_suggestion_prompt(current_context)
                    from llm_handler import LLMManager
                    llm_manager = LLMManager()
                    suggestions_json = llm_manager.generate(llm_client, prompt)
                    suggestions = extract_json_from_text(suggestions_json)

                    if suggestions:
                        st.success("✅ AI suggestions generated!")
                        
                        with st.expander("🤖 AI Suggestions", expanded=True):
                            for field_key, suggestion in suggestions.items():
                                if field_key != "reasoning":
                                    current_value = current_context.get(field_key, "")
                                    if current_value != suggestion and isinstance(suggestion, str) and suggestion.strip():
                                        field_label = next((label for key, label, _ in self.fields if key == field_key), field_key)
                                        
                                        st.write(f"**{field_label}:**")
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.write("Current:")
                                            st.info(current_value if current_value else "Empty")
                                        
                                        with col2:
                                            st.write("Suggested:")
                                            st.success(suggestion)
                                            
                                            if st.button(f"Apply", key=f"apply_{field_key}"):
                                                st.session_state.business_context[field_key] = suggestion
                                                st.success(f"Applied suggestion for {field_label}")
                                                st.rerun()
                            
                            if suggestions.get("reasoning"):
                                st.write("**AI Reasoning:**")
                                st.write(suggestions["reasoning"])
                        
                        if st.button("✅ Apply All Suggestions"):
                            for key, value in suggestions.items():
                                if key != "reasoning" and isinstance(value, str) and value.strip():
                                    st.session_state.business_context[key] = value
                            self._save_context_version("ai_enhancement")
                            st.success("All AI suggestions applied!")
                            st.rerun()
                    else:
                        st.error("❌ Could not generate AI suggestions. Please try again later.")
                except Exception as e:
                    st.error(f"❌ AI enhancement failed: {e}")

    def _display_version_management(self):
        """Display version management controls"""
        st.subheader("🔄 Version Management")
        
        project_id = st.session_state.get("current_project_id")
        if not project_id:
            st.info("Select a project to manage context versions")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Current Version"):
                self._save_context_version("manual")
                st.success("✅ Context version saved!")
        
        with col2:
            versions = self.db.get_all_context_versions(project_id)
            if versions:
                version_options = [
                    f"v{v['version_id']} - {v['source_type']} ({v['created_at'][:16]})"
                    for v in versions
                ]
                
                selected_version_str = st.selectbox(
                    "Load Previous Version",
                    options=version_options,
                    key="version_selector"
                )
                
                if st.button("🔄 Load Version"):
                    version_id = int(selected_version_str.split(" ")[0][1:])
                    selected_context = next((v['context'] for v in versions if v['version_id'] == version_id), None)
                    if selected_context:
                        st.session_state.business_context = selected_context
                        st.success("✅ Previous version loaded!")
                        st.rerun()

    def _update_validation_status(self):
        """Update validation status in session state for immediate feedback"""
        required_fields = [key for key, _, required in self.fields if required]
        completed_fields = []
        
        for key in required_fields:
            value = st.session_state.business_context.get(key, "")
            # Handle different value types safely
            if isinstance(value, str) and value.strip():
                completed_fields.append(key)
            elif isinstance(value, (list, dict)) and value:  # Non-empty list or dict
                completed_fields.append(key)
            elif value:  # Any other truthy value
                completed_fields.append(key)
        
        completion_rate = len(completed_fields) / len(required_fields) * 100 if required_fields else 100
        
        # Store validation status in session state for immediate access
        st.session_state.validation_status = {
            "completion_rate": completion_rate,
            "completed_fields": completed_fields,
            "missing_fields": [key for key in required_fields if key not in completed_fields]
        }

    def _display_validation_status(self):
        """Display validation status and next steps"""
        st.subheader("✅ Validation & Next Steps")
        
        # Use stored validation status or calculate fresh
        if "validation_status" in st.session_state:
            validation_status = st.session_state.validation_status
            completion_rate = validation_status["completion_rate"]
            completed_fields = validation_status["completed_fields"]
            missing_fields = validation_status["missing_fields"]
            required_fields_count = len(completed_fields) + len(missing_fields)
        else:
            required_fields = [key for key, _, required in self.fields if required]
            completed_fields = [
                key for key in required_fields 
                if st.session_state.business_context.get(key, "").strip()
            ]
            completion_rate = len(completed_fields) / len(required_fields) * 100 if required_fields else 100
            missing_fields = [key for key in required_fields if key not in completed_fields]
            required_fields_count = len(required_fields)
        
        st.progress(completion_rate / 100)
        st.write(f"**Completion:** {completion_rate:.0f}% ({len(completed_fields)}/{required_fields_count} required fields)")
        
        if missing_fields:
            missing_labels = [
                next(label for k, label, _ in self.fields if k == key)
                for key in missing_fields
            ]
            st.warning(f"**Missing required fields:** {', '.join(missing_labels)}")
        else:
            st.success("🎉 All required fields completed! You can proceed to Market Intelligence.")

    def _save_context_version(self, source_type: str = "manual"):
        """Save current context as a new version"""
        project_id = st.session_state.get("current_project_id")
        if project_id and st.session_state.business_context:
            self.db.save_context_version(
                project_id, 
                st.session_state.business_context, 
                source_type
            )

    def can_proceed_to_next_step(self):
        """Check if user can proceed to next step"""
        required_fields = [key for key, _, required in self.fields if required]
        if not required_fields:
            return True
        completed_fields = []
        for key in required_fields:
            value = st.session_state.business_context.get(key, "")
            # Handle different value types safely
            if isinstance(value, str) and value.strip():
                completed_fields.append(key)
            elif isinstance(value, (list, dict)) and value:  # Non-empty list or dict
                completed_fields.append(key)
            elif value:  # Any other truthy value
                completed_fields.append(key)
        return (len(completed_fields) / len(required_fields)) >= 0.8
