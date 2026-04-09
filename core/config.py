"""
Configuration constants and settings for Marketing AI
"""
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment via Pydantic"""
    
    # DB
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/marketingai"
    redis_url: str = "redis://localhost:6379/0"
    
    # LLM APIs
    groq_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    firecrawl_api_key: str = ""
    pandas_api_key: str = ""
    
    # Provider Endpoints
    provider_endpoints: Dict[str, str] = {
        "GROQ": "https://api.groq.com/openai/v1",
        "OPENAI": "https://api.openai.com/v1",
        "GEMINI": "https://generativelanguage.googleapis.com",
        "OLLAMA": "http://localhost:11434"
    }

    # Billing
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    pesapal_consumer_key: str = ""
    pesapal_consumer_secret: str = ""
    pesapal_sandbox: bool = True
    
    # Upload-Post
    upload_post_api_key: str = ""
    
    # Security
    webhook_signing_secret: str = "default_unsafe_secret"
    api_key_prefix: str = "mkai_"
    
    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 100
    similarity_threshold: float = 0.7

    # LLM Settings
    default_temperature: float = 0.3
    default_max_tokens: int = 4096

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Static Constants
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

    # File Processing
    MAX_FILE_SIZE_MB = 200
    SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md"]

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

    @staticmethod
    def get_api_key(provider: str) -> str:
        """Helper to get API key for a provider to ease migration"""
        settings = get_settings()
        key_map = {
            "GROQ": settings.groq_api_key,
            "OPENAI": settings.openai_api_key,
            "GEMINI": settings.gemini_api_key,
            "PANDASAI": settings.pandas_api_key,
            "FIRECRAWL": settings.firecrawl_api_key
        }
        return key_map.get(provider.upper(), "")

    @staticmethod
    def get_endpoint(provider: str) -> str:
        """Helper to get default endpoint for a provider"""
        settings = get_settings()
        return settings.provider_endpoints.get(provider.upper(), "")

# Singleton settings instance
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()
