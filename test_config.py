#!/usr/bin/env python3
"""
Test Configuration Setup

Quick test to verify that environment variables are properly loaded
and all components can access API keys from the .env file.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config


def test_configuration():
    """Test configuration loading"""
    print("\n[TEST] Testing Configuration Setup")
    print("=" * 40)
    
    # Print configuration status
    Config.print_status()
    
    # Test API key validation
    is_valid, missing_keys = Config.validate_required_keys()
    
    if is_valid:
        print("\n[PASS] Configuration test passed!")
        print("   All required API keys are configured.")
    else:
        print(f"\n[FAIL] Configuration test failed!")
        print(f"   Missing keys: {', '.join(missing_keys)}")
        print(f"   Please update your .env file with the missing API keys.")
    
    return is_valid


async def test_analyzer():
    """Test YouTube analyzer initialization"""
    print("\n[TEST] Testing YouTube Analyzer")
    print("-" * 30)
    
    try:
        from analyzer import YouTubeAnalyzer
        
        analyzer = YouTubeAnalyzer()
        print("[PASS] YouTube Analyzer initialized successfully")
        print(f"   Model: {analyzer.model}")
        print(f"   Rate limit: {analyzer.rate_limit_delay}s")
        print(f"   Max concurrent: {analyzer.max_concurrent}")
        return True
        
    except Exception as e:
        print(f"[FAIL] YouTube Analyzer failed: {e}")
        return False


async def test_rag_system():
    """Test RAG system initialization"""
    print("\n[TEST] Testing RAG System")
    print("-" * 20)
    
    try:
        from rag_pipeline.core import RAGSystem
        
        rag = RAGSystem()
        print("[PASS] RAG System created successfully")
        print(f"   Model: {rag.model}")
        print(f"   Collection: {rag.collection_name}")
        print(f"   Chunk size: {rag.chunk_size}")
        
        # Test initialization
        await rag.initialize()
        print("[PASS] RAG System initialized successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] RAG System failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n[START] YouTube RAG Analyzer - Configuration Test")
    print("=" * 50)
    
    # Test 1: Configuration
    config_ok = test_configuration()
    
    if not config_ok:
        print("\n[WARN] Skipping other tests due to configuration issues.")
        print("Please fix the configuration and run this test again.")
        return False
    
    # Test 2: Analyzer
    analyzer_ok = await test_analyzer()
    
    # Test 3: RAG System
    rag_ok = await test_rag_system()
    
    # Final results
    print("\n" + "=" * 50)
    print("[SUMMARY] Test Results Summary")
    print("=" * 50)
    print(f"Configuration: {'PASS' if config_ok else 'FAIL'}")
    print(f"YouTube Analyzer: {'PASS' if analyzer_ok else 'FAIL'}")
    print(f"RAG System: {'PASS' if rag_ok else 'FAIL'}")
    
    all_passed = config_ok and analyzer_ok and rag_ok
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Your system is ready to use.")
        print("\n[NEXT] Next steps:")
        print("   1. Run: python launch.py")
        print("   2. Or: streamlit run streamlit_app.py")
    else:
        print("\n[WARN] Some tests failed. Please check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)