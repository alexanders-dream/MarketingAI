import os
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    API_KEYS: Dict[str, str] = Field(
        default_factory=lambda: {
            "GROQ": os.getenv("GROQ_API_KEY", ""),
            "OPENAI": os.getenv("OPENAI_API_KEY", ""),
            "PANDASAI": os.getenv("PANDAS_API_KEY", "")
        }
    )
    RAG: Dict[str, Any] = {
        "CHUNK_SIZE": 10000,
        "CHUNK_OVERLAP": 300,
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "TOP_K_RESULTS": 5
    }
    MODELS: Dict[str, str] = {
        "DEFAULT_GROQ": "mixtral-8x7b-32768",
        "DEFAULT_OPENAI": "gpt-4-turbo"
    }
    TEMPERATURE: float = 0.7

    class Config:
        extra = 'allow'
        env_file = '.env'
        env_file_encoding = 'utf-8'

try:
    config = Settings()
except ValidationError as e:
    raise RuntimeError(f"Configuration error: {e}") from e

def get_api_key(provider: str) -> str:
    return config.API_KEYS.get(provider.upper(), "")