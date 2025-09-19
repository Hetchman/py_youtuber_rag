"""
Embeddings Module for YouTube RAG Analyzer

Provides embeddings functionality with multiple fallback options:
1. Nomic embeddings (preferred)
2. Google Generative AI embeddings  
3. SentenceTransformer embeddings (fallback)
"""

import os
from typing import List
import logging

logger = logging.getLogger(__name__)


class NomicEmbeddings:
    """Nomic embeddings wrapper"""
    
    def __init__(self, model: str = "nomic-embed-text-v1", nomic_api_key: str = None):
        self.model = model
        self.api_key = nomic_api_key or os.getenv('NOMIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("NOMIC_API_KEY is required")
            
        # Test connection
        try:
            import nomic
            if self.api_key:
                os.environ['NOMIC_API_KEY'] = self.api_key
        except ImportError:
            raise ImportError("nomic package not available. Install with: pip install nomic")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        import nomic
        
        try:
            response = nomic.embed.text(texts=texts, model=self.model)
            
            # Handle different response formats
            if isinstance(response, dict):
                embeddings = response.get('embeddings') or response.get('vectors')
            else:
                embeddings = response
                
            if embeddings is None:
                raise RuntimeError("No embeddings returned from Nomic API")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Nomic embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


def get_embeddings_provider(config=None):
    """
    Get the best available embeddings provider based on configuration and available API keys
    
    Returns initialized embeddings object
    """
    from config import Config
    
    # Try Nomic first if API key is available
    if Config.NOMIC_API_KEY and Config.NOMIC_API_KEY != "your_nomic_api_key_here":
        try:
            embeddings = NomicEmbeddings(nomic_api_key=Config.NOMIC_API_KEY)
            logger.info("Using Nomic embeddings")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to initialize Nomic embeddings: {e}")
    
    # Try LangChain Nomic integration
    try:
        from langchain_nomic import NomicEmbeddings as LCNomicEmbeddings
        if Config.NOMIC_API_KEY and Config.NOMIC_API_KEY != "your_nomic_api_key_here":
            embeddings = LCNomicEmbeddings(
                model="nomic-embed-text-v1.5",
                nomic_api_key=Config.NOMIC_API_KEY
            )
            logger.info("Using LangChain Nomic embeddings")
            return embeddings
    except Exception as e:
        logger.warning(f"LangChain Nomic embeddings not available: {e}")
    
    # Try Google Generative AI embeddings
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if Config.GOOGLE_API_KEY:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=Config.GOOGLE_API_KEY
            )
            logger.info("Using Google Generative AI embeddings")
            return embeddings
    except Exception as e:
        logger.warning(f"Google Generative AI embeddings not available: {e}")
    
    # Fallback to SentenceTransformer embeddings
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("Using SentenceTransformer embeddings as fallback")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize any embeddings: {e}")
        raise RuntimeError("No embeddings provider available")


if __name__ == "__main__":
    # Test embeddings
    embeddings = get_embeddings_provider()
    
    test_texts = ["Hello world", "This is a test"]
    result = embeddings.embed_documents(test_texts)
    
    print(f"Embedded {len(test_texts)} documents")
    print(f"Embedding dimension: {len(result[0])}")
    print(f"First embedding preview: {result[0][:5]}...")