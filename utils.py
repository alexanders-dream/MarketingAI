# utils.py
import requests
import streamlit as st

@st.cache_data
def fetch_models(provider, endpoint, api_key=None):
    if provider == "Groq":
        url = f"{endpoint}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return [model['id'] for model in response.json()['data']]
        else:
            return []
        
    elif provider == "OpenAI":
        url = f"{endpoint}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return [model['id'] for model in response.json()['data']]
        else:
            return []
        
    elif provider == "Ollama":
        url = f"{endpoint}/api/tags"  # Ollama endpoint to list models
        response = requests.get(url)
        if response.status_code == 200:
            return [model['name'] for model in response.json()['models']]
        else:
            return []
        
    return []