"""
Reusable UI components for Marketing AI v3
"""
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from io import BytesIO

from config import AppConfig
from llm_handler import LLMProviderHandler
from database import DatabaseManager


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
                index=0,
                key="task_select"
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
                    api_key = st.text_input(
                        f"{provider} API Key",
                        type="password",
                        value=AppConfig.get_api_key(provider),
                        help=f"Get your API key from {provider}'s dashboard",
                        key="api_key_input",
                        on_change=self._on_config_change
                    )

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
                    models = LLMProviderHandler.fetch_models(provider, endpoint, api_key)
                    st.session_state[cache_key] = models

            models = st.session_state.get(cache_key, [])

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
            else:
                st.warning(f"No models available for {provider}. Click '🔄 Refresh Models' to retry.")

        return model

    def _on_provider_change(self):
        """Handle provider change - clear model cache"""
        # Clear model cache when provider changes
        st.session_state.model = None
        # Clear any cached models for different providers
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("models_")]
        for key in keys_to_remove:
            del st.session_state[key]

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
            # Copy to clipboard (via text area)
            st.text_area(
                "Copy Content",
                value=content,
                height=100,
                help="Copy the content from this text area"
            )

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


class MarketingForm:
    """Handles the main marketing input form with comprehensive business information"""

    def __init__(self):
        self.all_fields = {
            "brand_description": {
                "label": "Brand Description",
                "placeholder": "Describe your brand, mission, values, and unique selling points",
                "required": True,
                "section": "brand"
            },
            "target_audience": {
                "label": "Target Audience",
                "placeholder": "Describe your ideal customers, demographics, psychographics, and characteristics",
                "required": True,
                "section": "audience"
            },
            "products_services": {
                "label": "Products/Services",
                "placeholder": "List and describe your main products and/or services offered",
                "required": True,
                "section": "products"
            },
            "marketing_goals": {
                "label": "Marketing Goals",
                "placeholder": "Identify your key marketing goals or objectives",
                "required": True,
                "section": "goals"
            },
            "existing_content": {
                "label": "Existing Content",
                "placeholder": "Summarize any existing marketing content, campaigns, or channels",
                "required": False,
                "section": "content"
            },
            "keywords": {
                "label": "SEO Keywords",
                "placeholder": "10-15 relevant keywords for marketing purposes (comma-separated)",
                "required": False,
                "section": "seo"
            },
            "suggested_topics": {
                "label": "Content Topics",
                "placeholder": "5-7 content topics relevant for your marketing strategy",
                "required": False,
                "section": "content"
            },
            "market_opportunities": {
                "label": "Market Opportunities",
                "placeholder": "Identify potential market opportunities, gaps, or areas for growth",
                "required": False,
                "section": "market"
            },
            "competitive_advantages": {
                "label": "Competitive Advantages",
                "placeholder": "Analyze and describe your competitive advantages and differentiators",
                "required": False,
                "section": "competition"
            },
            "customer_pain_points": {
                "label": "Customer Pain Points",
                "placeholder": "Identify customer pain points or problems that your business solves",
                "required": False,
                "section": "customers"
            }
        }

    def create_form(self, analysis_data: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
        """Create the comprehensive marketing input form"""
        with st.form("marketing_form"):
            st.header("📊 Business Information")

            # Get current task to determine which fields to show
            task = st.session_state.get("task", "Marketing Strategy")

            # Web scraping options
            self._create_web_scraping_section()

            # If we have analysis data, pre-populate fields
            default_values = analysis_data or {}

            # Group fields by sections for better UX
            sections = {
                "brand": ["brand_description"],
                "audience": ["target_audience"],
                "products": ["products_services"],
                "goals": ["marketing_goals"],
                "content": ["existing_content", "suggested_topics"],
                "seo": ["keywords"],
                "market": ["market_opportunities"],
                "competition": ["competitive_advantages"],
                "customers": ["customer_pain_points"]
            }

            # Determine which sections to show based on task
            visible_sections = self._get_visible_sections_for_task(task)

            form_data = {}

            for section_name, section_fields in sections.items():
                if section_name in visible_sections:
                    self._create_section(section_name, section_fields, default_values, form_data)

            # Task-specific additional fields
            self._create_task_specific_fields(task, default_values, form_data)

            # Submit button
            if st.form_submit_button(f"🚀 Generate {task}", type="primary"):
                # Validate required fields
                missing_required = self._validate_required_fields(form_data, task)
                if missing_required:
                    st.error(f"Please fill in required fields: {', '.join(missing_required)}")
                    return None

                return form_data

        return None

    def _create_web_scraping_section(self):
        """Create web scraping configuration section"""
        with st.expander("🌐 Web Research Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                use_web_scraping = st.checkbox(
                    "Enable Web Research",
                    value=True,
                    help="Automatically research industry trends and competitor data from the web",
                    key="use_web_scraping"
                )

            with col2:
                industry = st.text_input(
                    "Industry Sector",
                    placeholder="e.g., Technology, Healthcare, E-commerce",
                    help="Industry for web research (optional)",
                    key="industry_input"
                )

            if use_web_scraping:
                st.info("💡 Web research will enhance your analysis with current market trends, competitor insights, and industry data.")

    def _get_visible_sections_for_task(self, task: str) -> List[str]:
        """Determine which sections to show based on the selected task"""
        base_sections = ["brand", "audience", "products", "goals"]

        task_specific_sections = {
            "Marketing Strategy": ["content", "seo", "market", "competition", "customers"],
            "Market Analysis": ["market", "competition", "customers", "seo"],
            "Competitor Analysis": ["competition", "market", "customers"],
            "Post Composer": ["content", "seo", "audience"],
            "Content Calendar": ["content", "seo", "audience", "goals"],
            "Email Campaign": ["audience", "content", "goals"],
            "Social Media Strategy": ["content", "seo", "audience", "goals"],
            "SEO Strategy": ["seo", "content", "competition"],
            "Brand Guidelines": ["brand", "audience", "content"],
            "Ad Copy Generator": ["products", "audience", "goals"],
            "Landing Page": ["products", "audience", "goals", "brand"]
        }

        return base_sections + task_specific_sections.get(task, [])

    def _create_section(self, section_name: str, field_names: List[str],
                       default_values: Dict[str, str], form_data: Dict[str, str]):
        """Create a form section with multiple fields"""
        section_titles = {
            "brand": "🏢 Brand Identity",
            "audience": "👥 Target Audience",
            "products": "📦 Products & Services",
            "goals": "🎯 Marketing Goals",
            "content": "📝 Content & Marketing",
            "seo": "🔍 SEO & Keywords",
            "market": "📈 Market Analysis",
            "competition": "🏆 Competitive Landscape",
            "customers": "😊 Customer Insights"
        }

        st.subheader(section_titles.get(section_name, section_name))

        # Create columns for better layout
        num_fields = len(field_names)
        if num_fields == 1:
            cols = [st.container()]
        elif num_fields == 2:
            cols = st.columns(2)
        else:
            # For more fields, use a more compact layout
            cols = st.columns(2)

        for i, field_name in enumerate(field_names):
            field_config = self.all_fields[field_name]
            col_idx = i % len(cols)

            with cols[col_idx] if isinstance(cols[col_idx], st.delta_generator.DeltaGenerator) else cols[col_idx]:
                # Determine height based on field type
                height = 100 if field_name in ["keywords", "suggested_topics"] else 120

                value = st.text_area(
                    f"{field_config['label']}{' *' if field_config['required'] else ''}",
                    key=f"{field_name}_input",
                    value=default_values.get(field_name, ""),
                    height=height,
                    placeholder=field_config["placeholder"],
                    help=f"{'Required field' if field_config['required'] else 'Optional field'}"
                )

                form_data[field_name] = value

    def _create_task_specific_fields(self, task: str, default_values: Dict[str, str], form_data: Dict[str, str]):
        """Create task-specific additional fields"""
        if task == "Post Composer":
            st.subheader("🎨 Content Creation Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                post_type = st.selectbox(
                    "Content Type *",
                    options=["Instagram Post", "LinkedIn Post", "Twitter Thread", "Blog Post",
                            "Podcast Script", "Video Script", "Email Newsletter", "Media Brief"],
                    key="post_type_input",
                    help="Select the type of content to create"
                )

            with col2:
                tone = st.selectbox(
                    "Tone of Voice *",
                    options=["Professional", "Casual", "Inspirational", "Educational",
                            "Promotional", "Conversational", "Authoritative"],
                    key="tone_input",
                    help="Select the tone for the generated content"
                )

            with col3:
                content_length = st.selectbox(
                    "Content Length",
                    options=["Short (50-100 words)", "Medium (100-300 words)", "Long (300-500 words)"],
                    key="content_length_input",
                    help="Approximate desired content length"
                )

            # Topic selection from suggested topics
            suggested_topics = default_values.get("suggested_topics", "")
            if suggested_topics:
                topics_list = [topic.strip() for topic in suggested_topics.split(",") if topic.strip()]
                if topics_list:
                    selected_topic = st.selectbox(
                        "Content Topic (Optional)",
                        key="selected_topic_input",
                        options=["Choose from suggestions..."] + topics_list,
                        help="Select a suggested topic or leave blank for AI to choose"
                    )
                    if selected_topic != "Choose from suggestions...":
                        form_data["selected_topic"] = selected_topic

            form_data.update({
                "post_type": post_type,
                "tone": tone,
                "content_length": content_length
            })

        elif task in ["Email Campaign", "Social Media Strategy"]:
            st.subheader("📊 Campaign Details")

            col1, col2 = st.columns(2)

            with col1:
                campaign_goal = st.selectbox(
                    "Campaign Goal",
                    options=["Brand Awareness", "Lead Generation", "Sales Conversion",
                            "Customer Retention", "Community Building", "Thought Leadership"],
                    key="campaign_goal_input"
                )

            with col2:
                target_platforms = st.multiselect(
                    "Target Platforms",
                    options=["Email", "LinkedIn", "Instagram", "Twitter", "Facebook", "TikTok", "YouTube"],
                    key="target_platforms_input",
                    default=["Email"] if task == "Email Campaign" else ["LinkedIn", "Instagram"]
                )

            form_data.update({
                "campaign_goal": campaign_goal,
                "target_platforms": ", ".join(target_platforms)
            })

    def _validate_required_fields(self, form_data: Dict[str, str], task: str) -> List[str]:
        """Validate required fields based on task"""
        required_fields = ["brand_description", "target_audience", "products_services", "marketing_goals"]

        # Add task-specific required fields
        if task == "Post Composer":
            required_fields.extend(["post_type", "tone"])

        missing = []
        for field in required_fields:
            if not form_data.get(field, "").strip():
                field_config = self.all_fields.get(field)
                if field_config:
                    missing.append(field_config["label"])
                else:
                    missing.append(field.replace("_", " ").title())

        return missing
