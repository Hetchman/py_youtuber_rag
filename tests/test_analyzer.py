"""
Test Suite for YouTube RAG Analyzer

Tests for the Python implementation of the YouTube video analyzer
and RAG pipeline components.
"""

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from analyzer import YouTubeAnalyzer, AnalysisResult, BatchAnalysisResults
from rag_pipeline.core import RAGSystem


class TestYouTubeAnalyzer:
    """Test cases for YouTube analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            return YouTubeAnalyzer(api_key='test_key')
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.api_key == 'test_key'
        assert analyzer.model == 'gemini-2.5-flash'
        assert analyzer.rate_limit_delay == 2.0
        assert analyzer.max_concurrent == 3
    
    def test_analyzer_initialization_no_api_key(self):
        """Test analyzer initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                YouTubeAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_video_success(self, analyzer):
        """Test successful video analysis"""
        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = "This is a test analysis of the video content."
        
        with patch.object(analyzer, '_make_gemini_request', return_value=mock_response):
            result = await analyzer.analyze_video('https://www.youtube.com/watch?v=test123')
            
            assert isinstance(result, AnalysisResult)
            assert result.url == 'https://www.youtube.com/watch?v=test123'
            assert result.analysis == "This is a test analysis of the video content."
            assert result.error is None
            assert result.duration_seconds is not None
    
    @pytest.mark.asyncio
    async def test_analyze_video_timeout(self, analyzer):
        """Test video analysis timeout"""
        with patch.object(analyzer, '_make_gemini_request', side_effect=asyncio.TimeoutError()):
            result = await analyzer.analyze_video('https://www.youtube.com/watch?v=test123', timeout=1)
            
            assert isinstance(result, AnalysisResult)
            assert result.analysis is None
            assert "timed out" in result.error
    
    @pytest.mark.asyncio
    async def test_analyze_video_api_error(self, analyzer):
        """Test video analysis with API error"""
        with patch.object(analyzer, '_make_gemini_request', side_effect=Exception("API Error")):
            result = await analyzer.analyze_video('https://www.youtube.com/watch?v=test123')
            
            assert isinstance(result, AnalysisResult)
            assert result.analysis is None
            assert "API Error" in result.error
    
    @pytest.mark.asyncio
    async def test_batch_analysis(self, analyzer):
        """Test batch video analysis"""
        # Mock the analyze_video method
        async def mock_analyze_video(url, custom_prompt=None):
            return AnalysisResult(
                url=url,
                analysis=f"Analysis for {url}",
                duration_seconds=1.0
            )
        
        with patch.object(analyzer, 'analyze_video', side_effect=mock_analyze_video):
            with patch.object(analyzer, 'save_results', return_value=AsyncMock()):
                urls = [
                    'https://www.youtube.com/watch?v=test1',
                    'https://www.youtube.com/watch?v=test2'
                ]
                
                results = await analyzer.analyze_videos(urls, save_results=False)
                
                assert isinstance(results, BatchAnalysisResults)
                assert results.total_videos == 2
                assert len(results.results) == 2
                assert all(r.analysis is not None for r in results.results)
    
    @pytest.mark.asyncio
    async def test_save_and_load_results(self, analyzer):
        """Test saving and loading analysis results"""
        # Create test results
        test_results = BatchAnalysisResults(
            timestamp="2025-01-01T12:00:00",
            total_videos=1,
            results=[
                AnalysisResult(
                    url="https://www.youtube.com/watch?v=test123",
                    analysis="Test analysis content",
                    timestamp="2025-01-01T12:00:00"
                )
            ]
        )
        
        # Test saving to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer.results_dir = Path(temp_dir)
            
            # Save results
            filepath = await analyzer.save_results(test_results)
            assert filepath.exists()
            
            # Load results
            loaded_results = await analyzer.load_results(filepath)
            
            assert loaded_results.total_videos == test_results.total_videos
            assert len(loaded_results.results) == len(test_results.results)
            assert loaded_results.results[0].url == test_results.results[0].url
            assert loaded_results.results[0].analysis == test_results.results[0].analysis


class TestRAGSystem:
    """Test cases for RAG system"""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system instance for testing"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            return RAGSystem(api_key='test_key')
    
    def test_rag_initialization(self, rag_system):
        """Test RAG system initialization"""
        assert rag_system.api_key == 'test_key'
        assert rag_system.model == 'gemini-2.5-flash'
        assert rag_system.chunk_size == 1000
        assert rag_system.chunk_overlap == 200
        assert not rag_system._initialized
    
    def test_rag_initialization_no_api_key(self):
        """Test RAG system initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                RAGSystem()
    
    @pytest.mark.asyncio
    async def test_rag_initialization_process(self, rag_system):
        """Test RAG system initialization process"""
        # Mock all the components that get initialized
        with patch('langchain.chat_models.init_chat_model') as mock_llm:
            with patch.object(rag_system, '_setup_embeddings') as mock_embeddings:
                with patch.object(rag_system, '_setup_vector_store') as mock_vector_store:
                    with patch.object(rag_system, '_build_rag_graph') as mock_graph:
                        
                        await rag_system.initialize()
                        
                        assert rag_system._initialized
                        mock_llm.assert_called_once()
                        mock_embeddings.assert_called_once()
                        mock_vector_store.assert_called_once()
                        mock_graph.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_analysis_results(self, rag_system):
        """Test loading analysis results into RAG system"""
        # Create test analysis file
        test_data = {
            'timestamp': '2025-01-01T12:00:00',
            'total_videos': 1,
            'results': [
                {
                    'url': 'https://www.youtube.com/watch?v=test123',
                    'analysis': 'Test analysis content for RAG processing',
                    'timestamp': '2025-01-01T12:00:00',
                    'error': None,
                    'duration_seconds': 5.0
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file_path = f.name
        
        try:
            # Mock the add_documents method
            with patch.object(rag_system, 'add_documents', return_value=5) as mock_add_docs:
                chunks_added = await rag_system.load_analysis_results(temp_file_path)
                
                assert chunks_added == 5
                mock_add_docs.assert_called_once()
                
                # Check that the document was created correctly
                documents = mock_add_docs.call_args[0][0]
                assert len(documents) == 1
                assert documents[0].page_content == 'Test analysis content for RAG processing'
                assert documents[0].metadata['source'] == 'https://www.youtube.com/watch?v=test123'
                
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('GOOGLE_API_KEY'), reason="Requires Google API key")
    async def test_end_to_end_workflow(self):
        """Test complete workflow from analysis to RAG querying"""
        # This test requires a real API key and will make actual API calls
        # It's skipped by default unless GOOGLE_API_KEY is set
        
        # Initialize components
        analyzer = YouTubeAnalyzer()
        rag = RAGSystem(
            collection_name="test_collection",
            persist_directory="./test_chroma_db"
        )
        
        try:
            # Test video URL (short video to minimize API usage)
            test_url = "https://www.youtube.com/watch?v=WsEQjeZoEng"
            
            # Analyze video
            result = await analyzer.analyze_video(
                test_url,
                custom_prompt="Provide a brief summary of this video."
            )
            
            # Check analysis succeeded
            assert result.analysis is not None
            assert result.error is None
            
            # Initialize RAG system
            await rag.initialize()
            
            # Create a document from the analysis
            from langchain_core.documents import Document
            doc = Document(
                page_content=result.analysis,
                metadata={'source': test_url, 'type': 'test'}
            )
            
            # Add to RAG system
            chunks_added = await rag.add_documents([doc])
            assert chunks_added > 0
            
            # Query the RAG system
            response = await rag.ask_question("What is this video about?")
            
            assert response.answer
            assert len(response.answer) > 0
            
        finally:
            # Clean up test database
            import shutil
            test_db_path = Path("./test_chroma_db")
            if test_db_path.exists():
                shutil.rmtree(test_db_path)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])