"""
Basic Usage Examples for YouTube RAG Analyzer

This script demonstrates how to use the YouTube RAG Analyzer for:
1. Analyzing individual YouTube videos
2. Batch processing multiple videos
3. Setting up the RAG pipeline
4. Querying video content with natural language

Run this script to see the system in action!
"""

import asyncio
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from analyzer import YouTubeAnalyzer
from rag_pipeline import RAGSystem

# Sample YouTube videos for testing
SAMPLE_VIDEOS = [
    "https://www.youtube.com/watch?v=WsEQjeZoEng",  # Google I/O 2024 Keynote
    "https://www.youtube.com/watch?v=9hE5-98ZeCg",  # Another sample video
]

SAMPLE_QUESTIONS = [
    "What are the main topics covered in these videos?",
    "What new AI features were announced?",
    "Can you summarize the key takeaways?",
    "What technologies or products were mentioned?",
    "What are the implications for developers?"
]


async def example_single_video_analysis():
    """Example 1: Analyze a single YouTube video"""
    print("🎬 Example 1: Single Video Analysis")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = YouTubeAnalyzer()
        
        # Analyze single video
        video_url = SAMPLE_VIDEOS[0]
        print(f"Analyzing: {video_url}")
        
        result = await analyzer.analyze_video(
            video_url,
            custom_prompt="Provide a concise summary focusing on the main announcements and key features discussed."
        )
        
        if result.analysis:
            print("\n📊 Analysis Result:")
            print("-" * 30)
            print(result.analysis[:500] + "..." if len(result.analysis) > 500 else result.analysis)
            print(f"\n⏱️ Analysis took: {result.duration_seconds:.2f} seconds")
        else:
            print(f"❌ Analysis failed: {result.error}")
            
    except Exception as e:
        print(f"❌ Error in single video analysis: {e}")


async def example_batch_analysis():
    """Example 2: Batch process multiple videos"""
    print("\n\n🎬 Example 2: Batch Video Analysis")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = YouTubeAnalyzer(
            rate_limit_delay=1.0,  # Faster for demo
            max_concurrent=2
        )
        
        # Analyze multiple videos
        print(f"Analyzing {len(SAMPLE_VIDEOS)} videos in batch...")
        
        results = await analyzer.analyze_videos(
            SAMPLE_VIDEOS,
            custom_prompt="Extract the key points and main themes from this video.",
            save_results=True
        )
        
        print(f"\n📊 Batch Analysis Complete!")
        print(f"✅ Successful analyses: {len([r for r in results.results if r.analysis])}")
        print(f"❌ Failed analyses: {len([r for r in results.results if r.error])}")
        
        # Show first result preview
        if results.results and results.results[0].analysis:
            print(f"\n📄 Sample result preview:")
            print("-" * 30)
            preview = results.results[0].analysis[:300] + "..."
            print(preview)
            
    except Exception as e:
        print(f"❌ Error in batch analysis: {e}")


async def example_rag_setup_and_query():
    """Example 3: Setup RAG pipeline and query video content"""
    print("\n\n🧠 Example 3: RAG Pipeline Setup and Querying")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag = RAGSystem(
            collection_name="youtube_demo",
            persist_directory="./demo_chroma_db"
        )
        await rag.initialize()
        
        # Check if we have analysis results to load
        results_dir = Path("analysis-results")
        if results_dir.exists():
            # Find the most recent analysis file
            json_files = list(results_dir.glob("analysis-*.json"))
            if json_files:
                latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"📄 Loading analysis results from: {latest_file}")
                
                chunks_added = await rag.load_analysis_results(latest_file)
                print(f"✅ Added {chunks_added} chunks to RAG system")
                
                # Ask questions about the content
                print("\n🤔 Asking questions about the video content...")
                
                for i, question in enumerate(SAMPLE_QUESTIONS[:3], 1):  # Limit to 3 for demo
                    print(f"\n❓ Question {i}: {question}")
                    
                    response = await rag.ask_question(
                        question,
                        conversation_id=f"demo_conversation_{i}"
                    )
                    
                    print(f"💬 Answer: {response.answer[:300]}...")
                    if response.sources:
                        print(f"📚 Sources: {len(response.sources)} documents")
                
            else:
                print("⚠️ No analysis results found. Please run video analysis first.")
        else:
            print("⚠️ No analysis results directory found. Please run video analysis first.")
            
        # Show collection stats
        stats = await rag.get_collection_stats()
        print(f"\n📊 RAG Collection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error in RAG setup: {e}")


async def example_conversational_rag():
    """Example 4: Conversational RAG interaction"""
    print("\n\n💬 Example 4: Conversational RAG")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        await rag.initialize()
        
        # Check if we have data
        stats = await rag.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            print("⚠️ No documents in RAG system. Load analysis results first.")
            return
        
        # Simulate a conversation
        conversation_id = "demo_conversation"
        
        questions = [
            "What are the main topics discussed in the videos?",
            "Can you elaborate on the AI-related announcements?",
            "What should developers pay attention to?"
        ]
        
        print("🗣️ Starting conversational interaction...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n👤 Human ({i}): {question}")
            
            response = await rag.ask_question(question, conversation_id)
            
            # Truncate response for demo
            answer = response.answer
            if len(answer) > 200:
                answer = answer[:200] + "..."
            
            print(f"🤖 Assistant ({i}): {answer}")
            
    except Exception as e:
        print(f"❌ Error in conversational RAG: {e}")


async def main():
    """Run all examples"""
    print("🎉 YouTube RAG Analyzer - Example Usage")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key in a .env file or environment variable.")
        return
    
    # Run examples
    print("🔑 API Key: ✅ Found")
    print("\nRunning examples...")
    
    try:
        # Example 1: Single video analysis
        await example_single_video_analysis()
        
        # Example 2: Batch analysis
        await example_batch_analysis()
        
        # Example 3: RAG setup and querying
        await example_rag_setup_and_query()
        
        # Example 4: Conversational RAG
        await example_conversational_rag()
        
    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
    
    print("\n🎉 Examples completed!")
    print("\n📚 Next Steps:")
    print("1. Modify the SAMPLE_VIDEOS list with your own YouTube URLs")
    print("2. Customize the analysis prompts in analyzer.py")
    print("3. Explore the RAG pipeline configuration options")
    print("4. Build your own application using these components!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())