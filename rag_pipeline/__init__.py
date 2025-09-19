"""
RAG Pipeline Package for YouTube Video Analysis

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline
for processing and querying YouTube video analysis results using the latest
LangChain patterns and integrations.

Components:
- Core RAG system with LangGraph orchestration
- Nomic embeddings for high-quality vector representations
- Chroma vector database with persistence
- Conversational chat interface with memory
- Document processing and chunking optimized for video analysis
"""

from .core import RAGSystem
from .embeddings import NomicEmbeddings, get_embeddings_provider
from .vector_store import ChromaVectorStore
from .chat import ConversationalRAG, SimpleConversationalRAG

__version__ = "1.0.0"
__author__ = "YouTube RAG Analyzer"

__all__ = [
    "RAGSystem",
    "NomicEmbeddings",
    "get_embeddings_provider", 
    "ChromaVectorStore",
    "ConversationalRAG",
    "SimpleConversationalRAG"
]