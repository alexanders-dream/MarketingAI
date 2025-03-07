# config.py
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class AppConfig:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._init_config()
        return cls._instance
    
    @classmethod
    def _init_config(cls):
        cls.settings = {
            "API_KEYS": {
                "GROQ": os.getenv("GROQ_API_KEY"),
                "OPENAI": os.getenv("OPENAI_API_KEY")
            },
            "RAG": {
                "CHUNK_SIZE": 512,
                "CHUNK_OVERLAP": 64,
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
                "TOP_K_RESULTS": 3
            },
            "MODELS": {
                "DEFAULT_GROQ": "mixtral-8x7b-32768",
                "DEFAULT_OPENAI": "gpt-4-turbo"
            }
        }
    
    @classmethod
    def get(cls, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = cls.settings
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            return default

def get_api_key(provider: str) -> str:
    return AppConfig().get(f"API_KEYS.{provider.upper()}")