"""
Core RAG System Implementation

Modern RAG pipeline using LangChain, LangGraph, Gemini, Nomic, and Chroma
following the latest patterns from LangChain tutorials.

Features:
- LangGraph-based workflow orchestration  
- Gemini 2.5 Flash for generation
- Nomic embeddings for vector representations
- Chroma vector store with persistence
- Conversational memory and state management
- Document processing optimized for video analysis
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


@dataclass
class RAGResponse:
    """Response structure for RAG queries"""
    answer: str
    sources: List[Document]
    conversation_id: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class State(TypedDict):
    """State structure for LangGraph workflow"""
    question: str
    context: List[Document]
    answer: str


class RAGSystem:
    """
    Modern RAG System using LangGraph and latest LangChain patterns
    
    Implements a conversational RAG pipeline for YouTube video analysis results
    with state management, memory, and sophisticated retrieval strategies.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        nomic_api_key: Optional[str] = None,
        model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        collection_name: str = None,
        persist_directory: str = None
    ):
        """
        Initialize the RAG System
        
        Args:
            api_key: Google API key for Gemini (defaults to config)
            nomic_api_key: Nomic API key for embeddings (defaults to config)
            model: Gemini model to use (defaults to config)
            chunk_size: Size of document chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
            collection_name: Chroma collection name (defaults to config)
            persist_directory: Directory for Chroma persistence (defaults to config)
        """
        # API keys from config
        self.api_key = api_key or Config.GOOGLE_API_KEY
        self.nomic_api_key = nomic_api_key or Config.NOMIC_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY in .env file.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Configuration from config class
        self.model = model or Config.DEFAULT_MODEL
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = Path(persist_directory or Config.CHROMA_PERSIST_DIRECTORY)
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self.graph = None
        self.memory = None
        
        # Ensure directories exist
        self.persist_directory.mkdir(exist_ok=True)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all RAG components"""
        if self._initialized:
            return
        
        print("[INIT] Initializing RAG System...")
        
        # Initialize LLM
        print("[LLM] Setting up Gemini LLM...")
        self.llm = init_chat_model(
            self.model, 
            model_provider="google_genai",
            api_key=self.api_key
        )
        
        # Initialize embeddings
        print("[EMBED] Setting up embeddings...")
        await self._setup_embeddings()
        
        # Initialize vector store
        print("[VECTOR] Setting up vector store...")
        await self._setup_vector_store()
        
        # Initialize text splitter
        print("[SPLIT] Setting up text splitter...")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        
        # Initialize memory
        print("[MEMORY] Setting up memory...")
        self.memory = MemorySaver()
        
        # Build the RAG graph
        print("[GRAPH] Building RAG workflow...")
        await self._build_rag_graph()
        
        self._initialized = True
        print("[SUCCESS] RAG System initialized successfully!")
    
    async def _setup_embeddings(self):
        """Setup embeddings (Nomic or fallback)"""
        try:
            from .embeddings import get_embeddings_provider
            
            self.embeddings = get_embeddings_provider()
            print("Embeddings provider initialized successfully")
                
        except Exception as e:
            print(f"Error setting up embeddings: {e}")
            # Final fallback to sentence transformers
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print("Using SentenceTransformer embeddings as final fallback")
    
    async def _setup_vector_store(self):
        """Setup Chroma vector store"""
        try:
            from .vector_store import ChromaVectorStore
            
            self.vector_store = ChromaVectorStore(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
                embeddings=self.embeddings
            )
            print("Chroma vector store ready")
            
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            # Fallback to in-memory vector store
            from langchain_core.vectorstores import InMemoryVectorStore
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            print("Using in-memory vector store as fallback")
    
    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str) -> tuple[str, List[Document]]:
        """Retrieve documents relevant to a query"""
        retrieved_docs = self.vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    async def _build_rag_graph(self):
        """Build the LangGraph RAG workflow"""
        # Create tools
        tools = [self.retrieve]
        tools_node = ToolNode(tools)
        
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly"""
            llm_with_tools = self.llm.bind_tools(tools)
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def generate(state: MessagesState):
            """Generate answer using retrieved context"""
            # Get recent tool messages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            # Format context
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks about YouTube videos. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Keep the answer comprehensive but well-organized with clear headings. "
                "Always cite the source videos when providing information."
                "\n\n"
                f"{docs_content}"
            )
            
            # Prepare conversation messages
            conversation_messages = [
                message for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            
            # Generate response
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Build graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools_node)
        graph_builder.add_node("generate", generate)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"}
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        # Compile with memory
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    async def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of chunks added
        """
        if not self._initialized:
            await self.initialize()
        
        print(f"[DOC] Processing {len(documents)} documents...")
        
        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        # Add to vector store
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_store.add_documents(all_chunks)
        )
        
        print(f"[SUCCESS] Added {len(all_chunks)} chunks to vector store")
        return len(all_chunks)
    
    async def load_analysis_results(self, results_file: Union[str, Path]) -> int:
        """
        Load YouTube analysis results into the RAG system
        
        Args:
            results_file: Path to analysis results JSON file
            
        Returns:
            Number of chunks added
        """
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for result in data.get('results', []):
                if result.get('analysis'):
                    doc = Document(
                        page_content=result['analysis'],
                        metadata={
                            'source': result['url'],
                            'timestamp': result['timestamp'],
                            'type': 'youtube_analysis'
                        }
                    )
                    documents.append(doc)
            
            if documents:
                return await self.add_documents(documents)
            else:
                print("[WARN] No valid analysis results found in file")
                return 0
                
        except Exception as e:
            print(f"[ERROR] Error loading analysis results: {e}")
            raise
    
    async def ask_question(
        self,
        question: str,
        conversation_id: str = "default"
    ) -> RAGResponse:
        """
        Ask a question using the RAG system
        
        Args:
            question: Question to ask
            conversation_id: Conversation thread ID
            
        Returns:
            RAGResponse with answer and sources
        """
        if not self._initialized:
            await self.initialize()
        
        # Configuration for conversation thread
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Run the graph
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.graph.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config
            )
        )
        
        # Extract answer and sources
        answer = result["messages"][-1].content if result["messages"] else "No answer generated"
        
        # Get sources from tool messages
        sources = []
        for message in result["messages"]:
            if hasattr(message, 'artifact') and message.artifact:
                sources.extend(message.artifact)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id
        )
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # For Chroma
            if hasattr(self.vector_store, '_collection'):
                collection = self.vector_store._collection
                count = collection.count()
                return {
                    "total_documents": count,
                    "collection_name": self.collection_name,
                    "persist_directory": str(self.persist_directory)
                }
            else:
                # For other vector stores
                return {
                    "vector_store_type": type(self.vector_store).__name__,
                    "collection_name": self.collection_name
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        if not self._initialized:
            await self.initialize()
        
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.vector_store.similarity_search(query, k=k)
        )


# Convenience function for quick setup
async def create_rag_system(**kwargs) -> RAGSystem:
    """Create and initialize a RAG system"""
    rag = RAGSystem(**kwargs)
    await rag.initialize()
    return rag