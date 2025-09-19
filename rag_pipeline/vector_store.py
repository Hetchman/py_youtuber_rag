"""
Vector Store Module for YouTube RAG Analyzer

Provides vector storage functionality using Chroma with fallback options.
"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Chroma vector store wrapper with fallback capabilities"""
    
    def __init__(
        self,
        collection_name: str = "youtube_analysis",
        persist_directory: str = "./chroma_db",
        embeddings=None
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embeddings = embeddings
        self.vector_store = None
        self.chroma_client = None
        self._doc_count = 0
        
        # Ensure directory exists
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store with fallback"""
        try:
            # Try Chroma first
            from langchain_chroma import Chroma
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"Initialized Chroma vector store: {self.collection_name}")
            return
            
        except Exception as e:
            logger.warning(f"Failed to initialize Chroma: {e}")
            
            # Try legacy Chroma import
            try:
                from langchain_community.vectorstores import Chroma
                
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
                
                logger.info(f"Initialized legacy Chroma vector store: {self.collection_name}")
                return
                
            except Exception as e2:
                logger.warning(f"Legacy Chroma also failed: {e2}")
        
        # Fallback to in-memory vector store
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            logger.warning("Using in-memory vector store as fallback")
            
        except Exception as e:
            logger.error(f"All vector store options failed: {e}")
            raise RuntimeError("No vector store available")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_store.add_documents(batch)
                self._doc_count += len(batch)
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Persist if supported
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
                
            logger.info("Documents successfully added")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform similarity search"""
        try:
            if filter_dict:
                return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores"""
        try:
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                if filter_dict:
                    return self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
                else:
                    return self.vector_store.similarity_search_with_score(query, k=k)
            else:
                # Fallback: return documents with dummy scores
                docs = self.similarity_search(query, k=k, filter_dict=filter_dict)
                return [(doc, 1.0) for doc in docs]
                
        except Exception as e:
            logger.error(f"Error during scored similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if hasattr(self.vector_store, '_collection'):
                # Chroma store
                count = self.vector_store._collection.count()
            else:
                # In-memory store
                count = self._doc_count
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": str(self.persist_directory)
            }
    
    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            if hasattr(self.vector_store, 'delete_collection'):
                self.vector_store.delete_collection()
            
            # Also remove persist directory if it exists
            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory)
            
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection"""
        try:
            self.delete_collection()
            
            # Recreate directory
            self.persist_directory.mkdir(exist_ok=True)
            
            # Reinitialize vector store
            self._initialize_vector_store()
            self._doc_count = 0
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False


if __name__ == "__main__":
    # Test vector store
    from embeddings import get_embeddings_provider
    
    embeddings = get_embeddings_provider()
    vector_store = ChromaVectorStore(embeddings=embeddings)
    
    # Test documents
    test_docs = [
        Document(page_content="This is a test document", metadata={"source": "test1"}),
        Document(page_content="Another test document", metadata={"source": "test2"})
    ]
    
    vector_store.add_documents(test_docs)
    
    # Test search
    results = vector_store.similarity_search("test", k=2)
    print(f"Found {len(results)} results")
    
    # Test stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")