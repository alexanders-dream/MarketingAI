# AI Marketing Assistant

## Demo
Try the live demo here: [AI Marketing Assistant](https://marketingai-agent.streamlit.app/)

## Overview
The AI Marketing Assistant is a Streamlit-based application that leverages LangChain and AI model providers (Groq and Ollama) to generate marketing content and strategies. Users can upload business documents, which are processed using a Retrieval-Augmented Generation (RAG) system for more accurate and contextually relevant outputs.

## Features
- **AI Provider Selection:** Choose between Groq and Ollama as the AI backend.
- **Document Processing:** Supports PDF, DOCX, TXT, and MD files for extracting business details.
- **Marketing Content Generation:** Generates various types of marketing content including:
  - Marketing Strategy
  - Campaign Ideas
  - Social Media Content
  - SEO Optimization
  - Copywriting
- **Vector Store Integration:** Uses FAISS for efficient document retrieval.
- **Customizable AI Settings:** Adjust temperature, max tokens, and model selection.
- **Downloadable Output:** Generated content can be downloaded as a DOCX file.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python, LangChain
- **LLMs:** Groq, Ollama
- **Vector Database:** FAISS
- **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`

## Installation
### Prerequisites
- Python 3.8+
- Groq or Ollama API access (if required)
- Streamlit

### Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-marketing-assistant.git
   cd ai-marketing-assistant
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file and add the following:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OLLAMA_API_KEY=your_ollama_api_key
   ```
5. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Select an AI provider and configure API keys (if required).
2. Upload a business document.
3. The app extracts and autofills relevant details.
4. Choose a marketing task to generate content.
5. Review and edit the generated content.
6. Download the final content as a DOCX file.



## Contribution
Feel free to submit pull requests or report issues. Contributions are welcome!

## License
This project is licensed under the MIT License.

