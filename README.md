# Marketing AI v3 - Comprehensive Marketing Content Generation Platform

## 🎯 Overview

Marketing AI v3 is a sophisticated, AI-powered marketing content generation platform built with Streamlit. It combines advanced language models, market intelligence, and document processing to create comprehensive marketing strategies and content. The application features a streamlined 3-step workflow that guides users from business context setup through market analysis to content generation.

## ✨ Key Features

### 🤖 Multi-Provider AI Integration
- **Supported Providers:** Groq, Ollama, OpenAI, Google Gemini
- **Dynamic Model Switching:** Change AI models and providers on-the-fly
- **Configurable Settings:** Adjust temperature, max tokens, and other parameters
- **LLM Pool Management:** Efficient caching and reuse of AI clients

### 📊 3-Step Workflow Process
1. **Business Context Setup** - Define your brand, target audience, and goals
2. **Market Intelligence** - Comprehensive market analysis with web scraping
3. **Content Generation** - AI-powered marketing content creation

### 📄 Advanced Document Processing
- **Supported Formats:** PDF, DOCX, TXT, MD files
- **RAG System:** Retrieval-Augmented Generation for context-aware content
- **Vector Storage:** FAISS integration for efficient document retrieval
- **Smart Extraction:** Automatic business information extraction from documents

### 🎨 Marketing Content Types
- **Marketing Strategy** - Comprehensive strategic plans with KPIs and timelines
- **Campaign Strategy** - Creative campaign concepts with implementation details
- **Social Media Content Strategy** - Platform-specific content calendars and tactics
- **SEO Optimization Strategy** - Technical and content SEO recommendations
- **Post Composer** - Individual social media posts and blog content
- **Market Analysis** - In-depth market research and competitive intelligence

### 🏢 Market Intelligence Hub
- **Competitive Analysis** - Competitor profiling and positioning
- **Market Trends** - Industry insights and opportunity identification
- **Target Segmentation** - Detailed audience profiling and segmentation
- **Growth Projections** - Market size and growth forecasting
- **Web Scraping** - Guided research for comprehensive analysis

### 💾 Project Management
- **Session Management** - Persistent project data across sessions
- **Content History** - Track and reuse generated content
- **Export Capabilities** - Download content in multiple formats

## 🛠️ Tech Stack

### Core Technologies
- **Frontend:** Streamlit
- **Backend:** Python 3.8+
- **AI Framework:** LangChain
- **Vector Database:** FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

### AI Providers & Models
- **Groq:** Fast inference with Llama models
- **Ollama:** Local model execution
- **OpenAI:** GPT series models
- **Google Gemini:** Multimodal capabilities

### Document Processing
- **PDF Processing:** PyPDF2, unstructured
- **Office Documents:** python-docx
- **Text Processing:** Built-in Python libraries
- **Web Scraping:** crawl4ai for market research

### Data Visualization
- **Charts:** Plotly for interactive market analysis charts
- **Dashboards:** Custom market intelligence visualizations

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- API keys for desired AI providers (optional for Ollama)

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/alexanders-dream/MarketingAI.git
   cd MarketingAI
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

### Environment Configuration
Create a `.env` file with the following variables:
```env
# AI Provider API Keys (configure as needed)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Ollama configuration (if using local models)
OLLAMA_BASE_URL=http://localhost:11434
```

## 📖 Usage Guide

### Step 1: Business Context Setup
1. **Access the application** at `http://localhost:8501`
2. **Create a new project** or select an existing one
3. **Upload business documents** (PDF, DOCX, TXT, MD) for context
4. **Fill in business details:**
   - Company name and brand description
   - Industry and target audience
   - Products/services and marketing goals
   - Existing content and SEO keywords

### Step 2: Market Intelligence
1. **Run market analysis** with optional web scraping
2. **Review comprehensive insights:**
   - Competitive landscape analysis
   - Market trends and opportunities
   - Target audience segmentation
   - Growth projections and sizing
3. **Explore interactive dashboards** with charts and visualizations

### Step 3: Content Generation
1. **Select content type** from available marketing tasks
2. **Configure additional parameters** (tone, platform, topics)
3. **Generate content** using AI with your business context
4. **Review and edit** generated content
5. **Download** in your preferred format

### Advanced Features
- **Model Switching:** Change AI providers mid-session
- **Content Scoring:** Get performance predictions for generated content
- **Project Persistence:** Work continues across sessions
- **Batch Processing:** Generate multiple content pieces efficiently

## 🏗️ Architecture

### Application Structure
```
marketing-ai-v3/
├── main.py                 # Application entry point
├── config.py              # Configuration constants
├── session_manager.py     # Session and project management
├── llm_handler.py         # AI provider management
├── content_generator.py   # Content generation engine
├── market_analyzer.py     # Market intelligence analysis
├── market_intelligence_ui.py  # Market analysis UI components
├── ui_components.py       # Reusable UI components
├── document_processor.py  # Document processing utilities
├── database.py           # Data persistence layer
├── prompts.py            # AI prompt templates
├── utils.py              # Utility functions
└── requirements.txt      # Python dependencies
```

### Key Components
- **Unified App Class:** Single entry point replacing dual applications
- **LLM Manager:** Handles multiple AI providers with standardized interface
- **Content Generator:** Template-based content creation with context injection
- **Market Intelligence:** Web scraping and analysis capabilities
- **Session Management:** Persistent state across user sessions

## 🔧 Configuration

### AI Model Settings
- **Temperature:** Controls creativity (0.0-1.0)
- **Max Tokens:** Response length limits
- **Model Selection:** Choose from available models per provider

### Document Processing
- **Max File Size:** 200MB per document
- **Chunk Size:** 1000 characters for RAG processing
- **Supported Formats:** PDF, DOCX, TXT, MD

### Market Analysis
- **Analysis Depth:** Basic, Comprehensive, or Deep Dive
- **Web Scraping:** Optional guided research
- **Data Sources:** Industry reports, competitor analysis, trends

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes:** `git commit -m 'Add amazing feature'`
5. **Push to the branch:** `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the powerful AI orchestration framework
- **Streamlit** for the excellent web application framework
- **Sentence Transformers** for high-quality embeddings
- **FAISS** for efficient vector similarity search
- **Plotly** for interactive data visualizations

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/alexanders-dream/MarketingAI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/alexanders-dream/MarketingAI/discussions)
- **Documentation:** See individual module docstrings and inline comments

---

**Built with ❤️ for marketers who believe in the power of AI-driven creativity**
