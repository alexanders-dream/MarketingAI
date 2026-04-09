"""
Configuration constants and settings for Marketing AI v3
"""
import os
from typing import Dict, Any


class AppConfig:
    """Application configuration constants"""

    # Marketing Tasks
    MARKETING_TASKS = [
        "Market Analysis",
        "Marketing Strategy",
        "Campaign Strategy",
        "Social Media Content Strategy",
        "SEO Optimization Strategy",
        "Post Composer"
    ]

    # API Keys
    API_KEYS: Dict[str, str] = {
        "GROQ": os.getenv("GROQ_API_KEY", ""),
        "OPENAI": os.getenv("OPENAI_API_KEY", ""),
        "GEMINI": os.getenv("GEMINI_API_KEY", ""),
        "PANDASAI": os.getenv("PANDAS_API_KEY", ""),
        "FIRECRAWL": os.getenv("FIRECRAWL_API_KEY", "")
    }

    # Provider Endpoints
    PROVIDER_ENDPOINTS = {
        "GROQ": "https://api.groq.com/openai/v1",
        "OPENAI": "https://api.openai.com/v1",
        "GEMINI": "https://generativelanguage.googleapis.com",
        "OLLAMA": "http://localhost:11434"
    }

    # File Processing
    MAX_FILE_SIZE_MB = 200
    SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md"]

    # RAG Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 100
    SIMILARITY_THRESHOLD = 0.7

    # LLM Settings
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 4096

    # Business Context
    BUSINESS_CONTEXT_FIELDS = [
        ("company_name", "Company Name", True),
        ("industry", "Industry", True),
        ("target_audience", "Target Audience", True),
        ("products_services", "Products/Services", True),
        ("brand_description", "Brand Description", True),
        ("marketing_goals", "Marketing Goals", True),
        ("existing_content", "Existing Content", False),
        ("keywords", "SEO Keywords", False),
        ("market_opportunities", "Market Opportunities", False),
        ("competitive_advantages", "Competitive Advantages", False),
        ("customer_pain_points", "Customer Pain Points", False),
        ("suggested_topics", "Suggested Topics", False),
    ]

    @classmethod
    def get_api_key(cls, provider: str) -> str:
        """Get API key for a provider"""
        return cls.API_KEYS.get(provider.upper(), "")

    @classmethod
    def get_endpoint(cls, provider: str) -> str:
        """Get default endpoint for a provider"""
        return cls.PROVIDER_ENDPOINTS.get(provider.upper(), "")
