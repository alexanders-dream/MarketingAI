# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key(provider):
    key_name = f"{provider.upper()}_API_KEY"
    return os.getenv(key_name)