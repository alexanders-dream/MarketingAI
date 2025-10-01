"""
Marketing AI v3 - Unified Main Application
Single entry point replacing dual apps with comprehensive refactoring
"""
import streamlit as st
import logging
from typing import Dict, Any

from config import AppConfig
from session_manager import SessionManager
from llm_handler import LLMManager
from ui_components import SidebarManager, ProjectManager, BusinessContextManager, ContentDisplay
from market_intelligence_ui import MarketIntelligenceDashboard, MarketAnalysisWizard
from content_generator import ContentGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketingAIApp:
    """Unified Marketing AI Application"""
    
    def __init__(self):
        self.app_config = AppConfig()
        self.session_manager = SessionManager()
        self.llm_manager = LLMManager()
        
        # Initialize UI components
        self.sidebar_manager = SidebarManager(self.session_manager)
        self.project_manager = ProjectManager(self.session_manager)
        self.content_display = ContentDisplay()
        self.content_generator = ContentGenerator()
        
        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize centralized session state"""
        if "initialized" not in st.session_state:
            # Create a default project when the session starts
            project_id = self.session_manager.create_project("Default Project")
            st.session_state.update({
                "initialized": True,
                "current_project_id": project_id,
                "current_step": 0,
                "business_context": {},
                "market_analysis_results": None,
                "generated_content": None,
                "llm_pool": {},
                "current_llm_config": None
            })

    def run(self):
        """Main application flow"""
        st.set_page_config(
            page_title="🎯 Marketing AI v3",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Sidebar configuration
        config = self.sidebar_manager.create_sidebar()
        
        # Get LLM client
        llm_client = self._get_or_create_llm_client(config)
        if not llm_client:
            st.error("⚠️ Could not initialize AI model. Please check your configuration.")
            return
        
        # Show model switching status
        if st.session_state.get("model_changed", False):
            st.success(f"✅ Switched to {config['provider']} - {config['model']}")
            st.session_state.model_changed = False
        
        # Main header
        st.title("🎯 Marketing AI Assistant")
        st.markdown("*Comprehensive AI-powered marketing content generation with market intelligence*")
        
        # Main workflow
        self._handle_main_workflow(llm_client, config)

    def _get_or_create_llm_client(self, config: Dict[str, Any]):
        """Get LLM client from pool or create new one"""
        config_key = f"{config['provider']}_{config['model']}_{config.get('api_endpoint', '')}_{config.get('temperature', 0.3)}_{config.get('max_tokens', 4096)}"
        
        # Check pool - use standardized interface validation
        if config_key in st.session_state.llm_pool:
            client = st.session_state.llm_pool[config_key]
            # Validate client using standardized interface check
            if self._is_valid_llm_client(client):
                return client
        
        # Create new client
        try:
            client = self.llm_manager.get_client(
                provider=config["provider"],
                model=config["model"],
                api_key=config.get("api_key"),
                endpoint=config.get("api_endpoint"),
                temperature=config.get("temperature", 0.3),
                max_tokens=config.get("max_tokens", 4096)
            )
            
            if client:
                # Store in pool (limit to 5)
                st.session_state.llm_pool[config_key] = client
                if len(st.session_state.llm_pool) > 5:
                    oldest_key = next(iter(st.session_state.llm_pool))
                    del st.session_state.llm_pool[oldest_key]
                
                st.session_state.current_llm_config = config_key
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create LLM client: {str(e)}")
            return None

    def _is_valid_llm_client(self, client) -> bool:
        """Validate LLM client using standardized interface check"""
        try:
            # Check if client has basic functionality by attempting a simple operation
            # All LangChain chat models should have these basic attributes/methods
            return (hasattr(client, 'invoke') and 
                   hasattr(client, 'model_name') or hasattr(client, 'model') and
                   hasattr(client, 'temperature') or hasattr(client, 'max_tokens'))
        except Exception:
            return False

    def _handle_main_workflow(self, llm_client, config):
        """Handle the main 3-step workflow"""
        if not st.session_state.get("current_project_id"):
            return
        
        # Step navigation
        st.markdown("---")
        steps = ["1️⃣ Business Context", "2️⃣ Market Intelligence", "3️⃣ Content Generation"]
        
        # Step selector
        current_step = st.radio(
            "**Workflow Steps:**",
            options=range(len(steps)),
            format_func=lambda x: steps[x],
            index=st.session_state.get("current_step", 0),
            horizontal=True,
            key="step_selector"
        )
        
        st.session_state.current_step = current_step
        
        # Step content
        if current_step == 0:
            self._handle_business_context_step(llm_client)
        elif current_step == 1:
            self._handle_market_intelligence_step(llm_client)
        elif current_step == 2:
            self._handle_content_generation_step(llm_client, config)

    def _handle_business_context_step(self, llm_client):
        """Handle Step 1: Business Context"""
        st.markdown("---")
        
        # Initialize Business Context Manager
        bcm = BusinessContextManager(self.session_manager)
        bcm.display_context_manager(llm_client)
        
        # Navigation helper
        if bcm.can_proceed_to_next_step():
            st.success("✅ Ready to proceed to Market Intelligence!")
            if st.button("➡️ Continue to Market Intelligence", type="primary"):
                st.session_state.current_step = 1
                st.rerun()

    def _handle_market_intelligence_step(self, llm_client):
        """Handle Step 2: Market Intelligence"""
        st.markdown("---")
        
        # Check prerequisites
        if not st.session_state.get("business_context"):
            st.warning("⚠️ Please complete Business Context first")
            if st.button("⬅️ Back to Business Context"):
                st.session_state.current_step = 0
                st.rerun()
            return
        
        # Market Intelligence Dashboard
        dashboard = MarketIntelligenceDashboard()
        wizard = MarketAnalysisWizard()
        
        if not st.session_state.get("market_analysis_results"):
            # Run analysis wizard
            st.header("🔍 Market Analysis")
            st.info("Let's analyze your market, competitors, and opportunities.")
            
            # Show business context summary
            with st.expander("📋 Current Business Context", expanded=False):
                st.json(st.session_state.business_context)
            
            # Run analysis
            analysis_results = wizard.run_analysis_wizard(
                st.session_state.business_context, 
                llm_client
            )
            
            if analysis_results:
                st.session_state.market_analysis_results = analysis_results
                st.rerun()
        else:
            # Display dashboard
            analysis_results = st.session_state.market_analysis_results
            
            # Refresh option
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("🔄 Refresh Analysis"):
                    del st.session_state.market_analysis_results
                    st.rerun()
            
            with col3:
                if st.button("➡️ Continue to Content Generation", type="primary"):
                    st.session_state.current_step = 2
                    st.rerun()
            
            # Display dashboard
            dashboard.display_full_dashboard(analysis_results)

    def _handle_content_generation_step(self, llm_client, config):
        """Handle Step 3: Content Generation"""
        st.markdown("---")
        
        # Check prerequisites
        if not st.session_state.get("business_context"):
            st.warning("⚠️ Please complete Business Context first")
            return
        
        st.header("3️⃣ Content Generation")
        st.markdown("Generate marketing content based on your business context and market intelligence.")

        # Task selection
        task = st.selectbox("Select a marketing task:", AppConfig.MARKETING_TASKS)

        # Additional inputs based on task
        additional_inputs = self._get_additional_inputs(task)

        if st.button("🚀 Generate Content"):
            if not st.session_state.get("current_project_id"):
                st.warning("Please select or create a project first.")
                return

            with st.spinner(f"Generating {task}..."):
                try:
                    # Use the already configured llm_client from the sidebar
                    if not llm_client:
                        st.error("LLM client not available. Check your API key.")
                        return

                    # Combine business context and additional inputs
                    context = {**st.session_state.business_context, **additional_inputs}

                    # Generate content
                    generated_content = self.content_generator.generate_content(
                        llm_client,
                        task,
                        context
                    )

                    if generated_content:
                        st.session_state.generated_content = generated_content
                        st.session_state.current_task = task
                        
                        # Display and save content
                        self.content_display.show_generated_content(
                            generated_content,
                            task,
                            st.session_state.current_project_id
                        )
                        self.session_manager.save_content(
                            st.session_state.current_project_id,
                            task,
                            generated_content,
                            context
                        )
                        st.success("Content generated and saved successfully!")
                    else:
                        st.error("Content generation failed.")

                except Exception as e:
                    logger.error(f"Error in content generation: {e}")
                    st.error(f"An error occurred: {e}")

    def _get_additional_inputs(self, task: str) -> dict:
        """Get additional user inputs required for specific tasks."""
        inputs = {}
        if task in ["Social Media Content Strategy", "Post Composer"]:
            inputs["post_type"] = st.selectbox("Select Post Type:", ["Instagram", "LinkedIn", "Twitter", "Blog"])
        if task in ["Campaign Strategy", "Social Media Content Strategy", "Post Composer"]:
            inputs["tone"] = st.selectbox("Select Tone:", ["Formal", "Casual", "Humorous", "Inspirational"])
        
        # Special handling for Post Composer - topic selection
        if task == "Post Composer":
            # Get suggested topics from business context
            suggested_topics = st.session_state.business_context.get("suggested_topics", "")
            
            if suggested_topics:
                # Parse topics from comma-separated string
                topic_list = [topic.strip() for topic in suggested_topics.split(",") if topic.strip()]
                
                # Add "Custom Topic" and "Random Topic" options
                topic_options = topic_list + ["Custom Topic", "Random Topic"]
                
                selected_topic = st.selectbox(
                    "Select Topic:",
                    options=topic_options,
                    help="Choose a suggested topic, create a custom one, or get a random topic"
                )
                
                if selected_topic == "Custom Topic":
                    # Allow user to input custom topic
                    inputs["selected_topic"] = st.text_input("Enter custom topic:", placeholder="Your custom topic here")
                elif selected_topic == "Random Topic":
                    # Select a random topic from the list
                    import random
                    if topic_list:
                        inputs["selected_topic"] = random.choice(topic_list)
                    else:
                        inputs["selected_topic"] = "Industry insights and trends"
                else:
                    inputs["selected_topic"] = selected_topic
            else:
                # Fallback if no suggested topics available
                inputs["selected_topic"] = st.text_input("Enter topic:", placeholder="Topic for your post")
        
        return inputs


def main():
    """Main entry point"""
    try:
        app = MarketingAIApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()
