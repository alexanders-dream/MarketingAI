# utils.py
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

@st.cache_data
def fetch_models(provider, endpoint, api_key=None):
    try:
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
            
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {str(e)}")
        st.sidebar.warning(f"By default, Ollama binds to 127.0. 0.1 , which restricts access to local connections only. To allow external access, you must set the OLLAMA_HOST variable to 0.0. 0.0 , enabling the server to accept connections from any IP address.")
        return None
        
    #return []

# utils.py
class ProviderHandler:
    @staticmethod
    def create_client(provider, model, api_key, endpoint):
        
        providers = {
            "Groq": lambda: ChatGroq(
                model=model,
                api_key=api_key,
                base_url="https://api.groq.com/",
                temperature=0.7
            ),
            "OpenAI": lambda: ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=endpoint,
                temperature=0.7
            ),
            "Ollama": lambda: ChatOllama(
                model=model,
                base_url=endpoint,
                temperature=0.7
            )
        }
        return providers.get(provider)()