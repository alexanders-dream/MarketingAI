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
        "PANDASAI": os.getenv("PANDAS_API_KEY", "")
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

    # Database
    DATABASE_PATH = "marketing_ai.db"

    @classmethod
    def get_api_key(cls, provider: str) -> str:
        """Get API key for a provider"""
        return cls.API_KEYS.get(provider.upper(), "")

    @classmethod
    def get_endpoint(cls, provider: str) -> str:
        """Get default endpoint for a provider"""
        return cls.PROVIDER_ENDPOINTS.get(provider.upper(), "")


class DatabaseConfig:
    """Database schema and queries"""

    # Table schemas
    PROJECTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    CONTENT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS generated_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        task_type TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    )
    """

    USER_PREFERENCES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT DEFAULT 'default',
        key TEXT NOT NULL,
        value TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, key)
    )
    """
