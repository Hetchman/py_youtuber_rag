"""
Conversational Chat Module for YouTube RAG Analyzer

Provides conversational interface for the RAG system with memory management.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class ConversationalRAG:
    """Conversational RAG interface with memory management"""
    
    def __init__(self, rag_system):
        """
        Initialize conversational RAG
        
        Args:
            rag_system: RAGSystem instance
        """
        self.rag_system = rag_system
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.system_message = """You are a helpful AI assistant that answers questions about YouTube video content. 
        You have access to analyzed video transcripts and summaries. Always cite the source video when possible.
        Be concise but informative in your responses."""
    
    async def chat(
        self,
        message: str,
        conversation_id: str = "default",
        max_history: int = 10
    ) -> Dict[str, Any]:
        """
        Have a conversation with the RAG system
        
        Args:
            message: User message
            conversation_id: Unique conversation identifier
            max_history: Maximum number of messages to keep in history
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Initialize conversation if needed
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
            
            # Add user message to history
            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            self.conversations[conversation_id].append(user_msg)
            
            # Get response from RAG system
            rag_response = await self.rag_system.query(message, conversation_id=conversation_id)
            
            # Add AI response to history
            ai_msg = {
                "role": "assistant",
                "content": rag_response.answer,
                "sources": [doc.metadata for doc in rag_response.sources],
                "timestamp": datetime.now().isoformat()
            }
            self.conversations[conversation_id].append(ai_msg)
            
            # Trim conversation history if too long
            if len(self.conversations[conversation_id]) > max_history * 2:  # *2 for user+ai pairs
                self.conversations[conversation_id] = self.conversations[conversation_id][-max_history * 2:]
            
            return {
                "response": rag_response.answer,
                "sources": rag_response.sources,
                "conversation_id": conversation_id,
                "timestamp": ai_msg["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            
            error_response = {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "sources": [],
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
            
            return error_response
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation"""
        return self.conversations.get(conversation_id, [])
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs"""
        return list(self.conversations.keys())
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def clear_all_conversations(self):
        """Clear all conversations"""
        self.conversations.clear()
    
    def export_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Export a conversation for saving/backup"""
        if conversation_id not in self.conversations:
            return None
        
        return {
            "conversation_id": conversation_id,
            "messages": self.conversations[conversation_id],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """Import a conversation from backup"""
        try:
            conversation_id = conversation_data["conversation_id"]
            messages = conversation_data["messages"]
            
            self.conversations[conversation_id] = messages
            return True
            
        except Exception as e:
            logger.error(f"Error importing conversation: {e}")
            return False
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary statistics for a conversation"""
        if conversation_id not in self.conversations:
            return {}
        
        messages = self.conversations[conversation_id]
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        ai_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        return {
            "conversation_id": conversation_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "last_activity": messages[-1]["timestamp"] if messages else None,
            "first_activity": messages[0]["timestamp"] if messages else None
        }


class SimpleConversationalRAG:
    """Simplified conversational interface for basic use cases"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.conversation_history = []
    
    async def ask(self, question: str) -> str:
        """Simple question-answering interface"""
        try:
            response = await self.rag_system.query(question)
            
            # Add to simple history
            self.conversation_history.append({
                "question": question,
                "answer": response.answer,
                "timestamp": datetime.now().isoformat()
            })
            
            return response.answer
            
        except Exception as e:
            logger.error(f"Error in simple chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()


if __name__ == "__main__":
    # Test conversational interface
    print("Conversational RAG module loaded successfully")
    print("This module requires a RAGSystem instance to function.")
    print("Use: chat = ConversationalRAG(rag_system)")