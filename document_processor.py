"""
Document processing and RAG system for Marketing AI v3
"""
import tempfile
import logging
from typing import Optional, Tuple, List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

from config import AppConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, processing, and vector storage"""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

    def load_document(self, file_bytes: bytes, file_name: str) -> Tuple[List[Document], str]:
        """
        Load and extract text from uploaded document

        Args:
            file_bytes: Raw file bytes
            file_name: Original filename

        Returns:
            Tuple of (documents, extracted_text)
        """
        file_extension = file_name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension in ['docx', 'doc']:
                loader = Docx2txtLoader(temp_path)
            elif file_extension in ['txt', 'md']:
                loader = TextLoader(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            documents = loader.load()
            extracted_text = " ".join([doc.page_content for doc in documents])

            return documents, extracted_text

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents

        Args:
            documents: List of LangChain documents

        Returns:
            FAISS vector store
        """
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.DEFAULT_CHUNK_SIZE,
            chunk_overlap=AppConfig.DEFAULT_CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        vector_store = FAISS.from_documents(splits, self.embeddings)

        return vector_store

    def process_document(self, file_bytes: bytes, file_name: str) -> Tuple[Optional[FAISS], str]:
        """
        Complete document processing pipeline

        Args:
            file_bytes: Raw file bytes
            file_name: Original filename

        Returns:
            Tuple of (vector_store, extracted_text)
        """
        try:
            # Validate file size
            if len(file_bytes) > AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File size exceeds {AppConfig.MAX_FILE_SIZE_MB}MB limit")

            # Validate file type
            file_extension = file_name.split('.')[-1].lower()
            if file_extension not in AppConfig.SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Load and process document
            documents, extracted_text = self.load_document(file_bytes, file_name)
            vector_store = self.create_vector_store(documents)

            return vector_store, extracted_text

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return None, ""

    def search_similar(self, vector_store: FAISS, query: str, k: int = 3,
                      score_threshold: float = AppConfig.SIMILARITY_THRESHOLD) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance filtering

        Args:
            vector_store: FAISS vector store
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of (document, score) tuples
        """
        try:
            # Perform similarity search with scores
            docs_and_scores = vector_store.similarity_search_with_score(query, k=k)

            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in docs_and_scores
                if score >= score_threshold
            ]

            return filtered_results

        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []

    def get_relevant_context(self, vector_store: FAISS, query: str,
                           max_chunks: int = 3) -> str:
        """
        Get relevant context for a query

        Args:
            vector_store: FAISS vector store
            query: Search query
            max_chunks: Maximum number of chunks to return

        Returns:
            Concatenated relevant text
        """
        relevant_docs = self.search_similar(vector_store, query, k=max_chunks)

        if not relevant_docs:
            return ""

        # Extract and combine text from relevant documents
        context_parts = [doc.page_content for doc, _ in relevant_docs]
        return "\n\n".join(context_parts)
