"""
Configuration Management for YouTube RAG Analyzer

Centralizes environment variable loading and configuration management.
Ensures all components can access environment variables consistently.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Get the project root directory (where .env should be located)
PROJECT_ROOT = Path(__file__).parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env file
if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
    print(f"[CONFIG] Loaded environment variables from: {ENV_FILE_PATH}")
else:
    # Fallback: try to load from current directory
    load_dotenv()
    print("[WARN] .env file not found in project root, using default environment variables")


class Config:
    """Configuration class that provides easy access to environment variables"""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    NOMIC_API_KEY: str = os.getenv("NOMIC_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    
    # LangSmith Settings
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    
    # Application Settings
    MAX_CONCURRENT_ANALYSES: int = int(os.getenv("MAX_CONCURRENT_ANALYSES", "3"))
    DEFAULT_RATE_LIMIT_DELAY: float = float(os.getenv("DEFAULT_RATE_LIMIT_DELAY", "2.0"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector Store Settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "youtube_analysis")
    
    # Analysis Settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    @classmethod
    def validate_required_keys(cls) -> tuple[bool, list[str]]:
        """
        Validate that all required API keys are present
        
        Returns:
            tuple: (is_valid, list_of_missing_keys)
        """
        missing_keys = []
        
        if not cls.GOOGLE_API_KEY or cls.GOOGLE_API_KEY == "your_gemini_api_key_here":
            missing_keys.append("GOOGLE_API_KEY")
        
        return len(missing_keys) == 0, missing_keys
    
    @classmethod
    def get_api_status(cls) -> dict:
        """
        Get the status of all API keys
        
        Returns:
            dict: Status of each API key
        """
        return {
            "google_api_key": bool(cls.GOOGLE_API_KEY and cls.GOOGLE_API_KEY != "your_gemini_api_key_here"),
            "nomic_api_key": bool(cls.NOMIC_API_KEY and cls.NOMIC_API_KEY != "your_nomic_api_key_here"),
            "langsmith_api_key": bool(cls.LANGSMITH_API_KEY and cls.LANGSMITH_API_KEY != "your_langsmith_api_key_here"),
            "langsmith_tracing": cls.LANGSMITH_TRACING
        }
    
    @classmethod
    def print_status(cls):
        """Print configuration status"""
        print("\n" + "=" * 50)
        print("[CONFIG] YouTube RAG Analyzer Configuration")
        print("=" * 50)
        
        # API Keys Status
        print("\n[API] API Keys:")
        status = cls.get_api_status()
        print(f"  Google (Gemini): {'[OK]' if status['google_api_key'] else '[MISSING]'}")
        print(f"  Nomic:           {'[OK]' if status['nomic_api_key'] else '[OPTIONAL]'}")
        print(f"  LangSmith:       {'[OK]' if status['langsmith_api_key'] else '[OPTIONAL]'}")
        
        # Application Settings
        print(f"\n[SETTINGS] Application Settings:")
        print(f"  Model:              {cls.DEFAULT_MODEL}")
        print(f"  Max Concurrent:     {cls.MAX_CONCURRENT_ANALYSES}")
        print(f"  Rate Limit Delay:   {cls.DEFAULT_RATE_LIMIT_DELAY}s")
        print(f"  Chunk Size:         {cls.CHUNK_SIZE}")
        print(f"  Collection Name:    {cls.COLLECTION_NAME}")
        
        # Validation
        is_valid, missing = cls.validate_required_keys()
        if is_valid:
            print(f"\n[OK] Configuration is valid and ready to use!")
        else:
            print(f"\n[ERROR] Missing required API keys: {', '.join(missing)}")
            print(f"   Please update your .env file: {ENV_FILE_PATH}")
        
        print("=" * 50)


# Set LangSmith environment variables if configured
if Config.LANGSMITH_API_KEY and Config.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "youtube-rag-analyzer"


# Export commonly used values
__all__ = [
    "Config",
    "PROJECT_ROOT",
    "ENV_FILE_PATH"
]


# Print status if run directly
if __name__ == "__main__":
    Config.print_status()