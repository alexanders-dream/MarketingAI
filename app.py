"""
Marketing AI v3 - Main Application
"""
import streamlit as st
import logging
from typing import Dict, Any

from config import AppConfig
from database import DatabaseManager
from document_processor import DocumentProcessor
from llm_handler import LLMManager
from market_analyzer import MarketAnalyzer
from content_generator import ContentGenerator, ContentPerformanceScorer
from image_generator import GeminiImageGenerator
from ui_components import (
    SidebarManager, ProjectManager, DocumentUploader,
    ContentDisplay, MarketingForm
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketingAIApp:
    """Main application class"""

    def __init__(self):
        self.app_config = AppConfig()
        # Initialize core components
        self.db = DatabaseManager()
        self.doc_processor = DocumentProcessor()
        self.llm_manager = LLMManager()
        self.market_analyzer = MarketAnalyzer()
        self.content_generator = ContentGenerator()
        self.performance_scorer = ContentPerformanceScorer()

        # Initialize UI components
        self.sidebar_manager = SidebarManager()
        self.project_manager = ProjectManager(self.db)
        self.document_uploader = DocumentUploader()
        self.content_display = ContentDisplay()
        self.marketing_form = MarketingForm()

        # Initialize optional components
        self.image_generator = None

        # Session state
        self._initialize_session_state()

        # LLM client pool for smooth switching
        self._initialize_llm_pool()

    def _initialize_llm_pool(self):
        """Initialize LLM client pool for smooth switching"""
        if "llm_pool" not in st.session_state:
            st.session_state.llm_pool = {}
        if "current_llm_config" not in st.session_state:
            st.session_state.current_llm_config = None

    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "initialized" not in st.session_state:
            st.session_state.update({
                "initialized": True,
                "current_project_id": None,
                "vector_store": None,
                "analysis_data": None,
                "generated_content": None,
                "processing_done": False,
                "last_uploaded_file": None,
                "task": self.app_config.MARKETING_TASKS[0]
            })

    def run(self):
        """Main application flow with smooth model switching"""
        st.set_page_config(
            page_title="Marketing AI v3",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Sidebar configuration
        config = self.sidebar_manager.create_sidebar()

        # Get or create LLM client from pool for smooth switching
        llm_client = self._get_or_create_llm_client(config)

        if not llm_client:
            st.error("Failed to initialize AI model. Please check your configuration.")
            return

        # Show model switching status if changed
        if st.session_state.get("model_changed", False):
            st.success(f"✅ Switched to {config['provider']} - {config['model']}")
            st.session_state.model_changed = False

        # Main content area
        st.title(f"📋 {config['task']} AI Generator")

        # Project management
        self._handle_project_management()

        # Document upload and processing
        self._handle_document_processing(llm_client)

        # Main workflow based on task
        self._handle_task_workflow(llm_client, config)

    def _get_or_create_llm_client(self, config: Dict[str, Any]):
        """Get LLM client from pool or create new one for smooth switching"""
        # Create a unique key for this configuration
        config_key = f"{config['provider']}_{config['model']}_{config['api_endpoint']}_{config['temperature']}_{config['max_tokens']}"

        # Check if we already have this client in the pool
        if config_key in st.session_state.llm_pool:
            client = st.session_state.llm_pool[config_key]
            # Verify client is still valid (basic check)
            if hasattr(client, '_client') or hasattr(client, 'model_name') or hasattr(client, 'model'):
                return client

        # Create new client and add to pool
        try:
            client = self.llm_manager.get_client(
                provider=config["provider"],
                model=config["model"],
                api_key=config["api_key"],
                endpoint=config["api_endpoint"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )

            if client:
                # Store in pool with limited size (keep last 5)
                st.session_state.llm_pool[config_key] = client
                if len(st.session_state.llm_pool) > 5:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(st.session_state.llm_pool))
                    del st.session_state.llm_pool[oldest_key]

                # Update current config
                st.session_state.current_llm_config = config_key

            return client

        except Exception as e:
            logger.error(f"Failed to create LLM client: {str(e)}")
            return None

    def _handle_project_management(self):
        """Handle project selection and creation"""
        st.header("📁 Project Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            project_id = self.project_manager.create_project_selector()

        with col2:
            new_project_id = self.project_manager.create_project_form()

        # Set current project
        current_project_id = project_id or new_project_id
        if current_project_id:
            st.session_state.current_project_id = current_project_id
            self.project_manager.show_project_info(current_project_id)

    def _handle_document_processing(self, llm_client):
        """Handle document upload and analysis"""
        st.header("📄 Document Analysis")

        # Document uploader
        upload_result = self.document_uploader.create_uploader()

        if upload_result:
            uploaded_file_bytes, file_name = upload_result

            if ("last_uploaded_file" not in st.session_state or
                st.session_state.last_uploaded_file != file_name):
                st.session_state.processing_done = False
                st.session_state.last_uploaded_file = file_name

            if st.button("🔍 Extract Business Insights"):
                st.session_state.processing_done = False

            if not st.session_state.processing_done:
                with st.spinner("Processing document and extracting insights..."):
                    self._process_document_and_analyze(llm_client, uploaded_file_bytes, file_name)

    def _process_document_and_analyze(self, llm_client, file_bytes: bytes, file_name: str):
        """Process uploaded document and perform market analysis"""
        try:
            vector_store, _ = self.doc_processor.process_document(file_bytes, file_name)

            if vector_store:
                st.session_state.vector_store = vector_store
                analysis_data = self.market_analyzer.generate_market_analysis(llm_client, vector_store)
                st.session_state.analysis_data = analysis_data
                self._display_analysis_results(analysis_data)
                st.session_state.processing_done = True
                st.success("✅ Document processed and insights extracted successfully!")
            else:
                st.error("Failed to process document. Please check the file format and try again.")

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            st.error(f"Error processing document: {str(e)}")

    def _display_analysis_results(self, analysis_data: Dict[str, str]):
        """Display market analysis results"""
        st.subheader("📊 Extracted Business Insights")

        # Create tabs for different insights
        tabs = st.tabs([
            "Brand", "Audience", "Products", "Goals",
            "Content", "Keywords", "Topics", "Opportunities"
        ])

        tab_data = [
            ("brand_description", "Brand Description"),
            ("target_audience", "Target Audience"),
            ("products_services", "Products & Services"),
            ("marketing_goals", "Marketing Goals"),
            ("existing_content", "Existing Content"),
            ("keywords", "Keywords"),
            ("suggested_topics", "Suggested Topics"),
            ("market_opportunities", "Market Opportunities")
        ]

        for i, (key, label) in enumerate(tab_data):
            with tabs[i]:
                content = analysis_data.get(key, "Not available")
                if key in ["keywords", "suggested_topics"]:
                    # Format as list
                    items = [item.strip() for item in content.split(",") if item.strip()]
                    for item in items:
                        st.write(f"• {item}")
                else:
                    st.write(content)

    def _handle_task_workflow(self, llm_client, config: Dict[str, Any]):
        """Handle the main task workflow"""
        task = config["task"]
        st.header(f"🎯 {task}")

        # Get analysis data for pre-population
        analysis_data = st.session_state.get("analysis_data")

        # Create marketing form
        form_data = self.marketing_form.create_form(analysis_data)

        if form_data and llm_client:
            with st.spinner(f"Generating {task}..."):
                # Generate content based on task
                if task == "Market Analysis":
                    content = self._generate_market_analysis_content(llm_client, form_data)
                else:
                    content = self.content_generator.generate_content(llm_client, task, form_data)

                if content:
                    st.session_state.generated_content = content

                    # Display content
                    self.content_display.show_generated_content(content, task, st.session_state.current_project_id)

                    # Save to database if project selected
                    if st.session_state.current_project_id:
                        self._save_content_to_project(content, task, form_data)

                    # Performance scoring for content tasks
                    if task in ["Post Composer", "Social Media Content Strategy", "SEO Optimization Strategy"]:
                        self._show_content_performance(llm_client, content, task)

                    # Image generation for visual content
                    if task == "Post Composer":
                        self._generate_content_images(content, form_data)

    def _generate_market_analysis_content(self, llm_client, form_data: Dict[str, str]) -> str:
        """Generate comprehensive market analysis"""
        return self.market_analyzer.generate_market_strategy(llm_client, form_data)

    def _save_content_to_project(self, content: str, task_type: str, metadata: Dict[str, Any]):
        """Save generated content to project"""
        try:
            project_id = st.session_state.current_project_id
            self.db.save_content(project_id, task_type, content, metadata)
            st.success("✅ Content saved to project!")
        except Exception as e:
            logger.error(f"Failed to save content: {str(e)}")
            st.error("Failed to save content to project.")

    def _show_content_performance(self, llm_client, content: str, content_type: str):
        """Show content performance analysis"""
        with st.expander("📈 Content Performance Analysis", expanded=False):
            score_data = self.performance_scorer.score_content(llm_client, content, content_type)
            self.content_display.show_performance_score(score_data)

    def _generate_content_images(self, content: str, form_data: Dict[str, str]):
        """Generate images for content"""
        with st.expander("🖼️ Generate Images", expanded=False):
            if not self.image_generator:
                self.image_generator = GeminiImageGenerator()

            # Image generation options
            col1, col2 = st.columns(2)

            with col1:
                aspect_ratio = st.selectbox(
                    "Aspect Ratio",
                    options=["1:1", "16:9", "4:3", "9:16"],
                    help="Choose the image aspect ratio"
                )

                style = st.selectbox(
                    "Style",
                    options=["natural", "vivid", "corporate", "creative", "social"],
                    help="Choose the image style"
                )

            with col2:
                prompt = st.text_area(
                    "Image Description",
                    value=f"Visual representation of: {content[:100]}...",
                    height=100,
                    help="Describe the image you want to generate"
                )

            if st.button("🎨 Generate Image"):
                with st.spinner("Generating image..."):
                    image_bytes = self.image_generator.generate_image(
                        prompt, aspect_ratio, style
                    )

                    if image_bytes:
                        st.image(image_bytes, caption="Generated Image", use_column_width=True)

                        # Download button
                        st.download_button(
                            label="💾 Download Image",
                            data=image_bytes,
                            file_name="generated_image.jpg",
                            mime="image/jpeg"
                        )
                    else:
                        st.error("Failed to generate image. Please check your Gemini API key.")


def main():
    """Application entry point"""
    app = MarketingAIApp()
    app.run()


if __name__ == "__main__":
    main()
