"""
Streamlit Web UI for YouTube RAG Analyzer

A comprehensive web interface for analyzing YouTube videos and querying content
using the RAG pipeline. Provides both video analysis and conversational chat
capabilities with a modern, user-friendly design.

Features:
- YouTube video analysis with custom prompts
- Batch processing interface
- RAG-powered question answering
- Conversation history management
- Analysis results visualization
- Real-time progress tracking
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
from streamlit_chat import message
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from analyzer import YouTubeAnalyzer, AnalysisResult, BatchAnalysisResults
from rag_pipeline.core import RAGSystem
from config import Config, PROJECT_ROOT
from utils.helpers import (
    is_youtube_url, validate_youtube_urls, format_duration, 
    truncate_text, extract_key_phrases, ColoredOutput
)

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'current_conversation_id' not in st.session_state:
        st.session_state.current_conversation_id = f"conv_{int(time.time())}"

# API Key validation
def check_api_keys():
    """Check if required API keys are configured"""
    is_valid, missing_keys = Config.validate_required_keys()
    
    if not is_valid:
        st.error(f"🔑 **Missing API Keys: {', '.join(missing_keys)}**")
        st.markdown(f"""
        <div class="error-message">
        <strong>Setup Required:</strong><br>
        1. Edit the <code>.env</code> file in: <code>{PROJECT_ROOT}</code><br>
        2. Add your Google API key: <code>GOOGLE_API_KEY=your_key_here</code><br>
        3. Get your API key from: <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    return True

# Initialize systems
@st.cache_resource
def get_analyzer():
    """Get or create YouTube analyzer instance"""
    try:
        return YouTubeAnalyzer(
            rate_limit_delay=Config.DEFAULT_RATE_LIMIT_DELAY,
            max_concurrent=Config.MAX_CONCURRENT_ANALYSES
        )
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {e}")
        return None

@st.cache_resource
def get_rag_system():
    """Get or create RAG system instance"""
    try:
        return RAGSystem(
            collection_name="youtube_streamlit",
            persist_directory="./streamlit_chroma_db"
        )
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

# Async wrapper for Streamlit
def run_async(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Video Analysis Functions
def analyze_videos_interface():
    """Interface for video analysis"""
    st.header("🎥 YouTube Video Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Video", "Multiple Videos", "Upload URL List"]
    )
    
    videos_to_analyze = []
    
    if input_method == "Single Video":
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        if video_url:
            videos_to_analyze = [video_url]
    
    elif input_method == "Multiple Videos":
        st.markdown("**Enter one URL per line:**")
        video_urls_text = st.text_area(
            "YouTube URLs:",
            height=150,
            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=..."
        )
        if video_urls_text:
            videos_to_analyze = [url.strip() for url in video_urls_text.split('\n') if url.strip()]
    
    elif input_method == "Upload URL List":
        uploaded_file = st.file_uploader(
            "Upload text file with URLs (one per line):",
            type=['txt', 'csv']
        )
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            videos_to_analyze = [url.strip() for url in content.split('\n') if url.strip()]
    
    # Validate URLs
    if videos_to_analyze:
        validation = validate_youtube_urls(videos_to_analyze)
        
        if validation['invalid_count'] > 0:
            st.warning(f"⚠️ Found {validation['invalid_count']} invalid URLs:")
            for url in validation['invalid_urls']:
                st.text(f"❌ {url}")
        
        if validation['valid_count'] > 0:
            st.success(f"✅ Found {validation['valid_count']} valid YouTube URLs")
            videos_to_analyze = validation['normalized_urls']
    
    # Custom prompt
    with st.expander("🎯 Custom Analysis Prompt (Optional)", expanded=False):
        custom_prompt = st.text_area(
            "Enter custom prompt:",
            placeholder="Analyze this video focusing on...",
            height=100
        )
    
    # Analysis settings
    col1, col2 = st.columns(2)
    with col1:
        rate_limit = st.slider("Rate limit (seconds between requests):", 1.0, 5.0, 2.0, 0.5)
    with col2:
        max_concurrent = st.slider("Max concurrent analyses:", 1, 5, 2)
    
    # Analyze button
    if st.button("🚀 Start Analysis", disabled=not videos_to_analyze):
        if not check_api_keys():
            return
        
        analyzer = get_analyzer()
        if not analyzer:
            return
        
        # Update analyzer settings
        analyzer.rate_limit_delay = rate_limit
        analyzer.max_concurrent = max_concurrent
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        try:
            # Run analysis
            status_text.text("🔄 Starting analysis...")
            
            async def run_analysis():
                return await analyzer.analyze_videos(
                    videos_to_analyze,
                    custom_prompt=custom_prompt if custom_prompt else None,
                    save_results=True
                )
            
            # Execute analysis
            batch_results = run_async(run_analysis())
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.analysis_results = batch_results.results
            
            # Display results
            display_analysis_results(batch_results)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            status_text.text("❌ Analysis failed")

def display_analysis_results(batch_results: BatchAnalysisResults):
    """Display analysis results with metrics and visualizations"""
    st.header("📊 Analysis Results")
    
    # Summary metrics
    successful = len([r for r in batch_results.results if r.analysis])
    failed = len(batch_results.results) - successful
    total_time = sum(r.duration_seconds or 0 for r in batch_results.results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Videos", batch_results.total_videos)
    with col2:
        st.metric("Successful", successful, delta=f"+{successful}")
    with col3:
        st.metric("Failed", failed, delta=f"-{failed}" if failed > 0 else "0")
    with col4:
        st.metric("Total Time", format_duration(total_time))
    
    # Results visualization
    if batch_results.results:
        # Create dataframe for visualization
        results_data = []
        for i, result in enumerate(batch_results.results):
            results_data.append({
                'Video': f"Video {i+1}",
                'Status': 'Success' if result.analysis else 'Failed',
                'Duration (s)': result.duration_seconds or 0,
                'URL': result.url
            })
        
        df = pd.DataFrame(results_data)
        
        # Status distribution chart
        status_counts = df['Status'].value_counts()
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Analysis Status Distribution",
            color_discrete_map={'Success': '#4ECDC4', 'Failed': '#FF6B6B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Duration analysis
        if any(df['Duration (s)'] > 0):
            fig_bar = px.bar(
                df,
                x='Video',
                y='Duration (s)',
                title="Analysis Duration by Video",
                color='Status',
                color_discrete_map={'Success': '#4ECDC4', 'Failed': '#FF6B6B'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Individual results
    st.subheader("📋 Individual Results")
    
    for i, result in enumerate(batch_results.results):
        with st.expander(f"🎬 Video {i+1}: {truncate_text(result.url, 50)}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**URL:** {result.url}")
                st.markdown(f"**Timestamp:** {result.timestamp}")
                if result.duration_seconds:
                    st.markdown(f"**Duration:** {format_duration(result.duration_seconds)}")
            
            with col2:
                if result.analysis:
                    st.success("✅ Success")
                else:
                    st.error("❌ Failed")
                    if result.error:
                        st.error(f"Error: {result.error}")
            
            if result.analysis:
                st.markdown("**Analysis:**")
                st.markdown(result.analysis)
                
                # Extract key phrases
                key_phrases = extract_key_phrases(result.analysis, max_phrases=5)
                if key_phrases:
                    st.markdown("**Key Phrases:**")
                    st.write(", ".join(f"`{phrase}`" for phrase in key_phrases))

# RAG Interface Functions
def rag_interface():
    """Interface for RAG-powered questioning"""
    st.header("🧠 Ask Questions About Videos")
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        with st.spinner("🔄 Initializing RAG system..."):
            rag_system = get_rag_system()
            if rag_system:
                try:
                    run_async(rag_system.initialize())
                    st.session_state.rag_system = rag_system
                    st.success("✅ RAG system initialized!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize RAG system: {e}")
                    return
            else:
                return
    
    rag_system = st.session_state.rag_system
    
    # Load analysis results into RAG
    st.subheader("📚 Load Analysis Data")
    
    # Check for existing analysis results
    results_dir = Path("analysis-results")
    if results_dir.exists():
        json_files = list(results_dir.glob("analysis-*.json"))
        if json_files:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_file = st.selectbox(
                    "Select analysis file to load:",
                    options=json_files,
                    format_func=lambda x: f"{x.name} ({x.stat().st_mtime})"
                )
            
            with col2:
                if st.button("📥 Load Data"):
                    try:
                        with st.spinner("Loading analysis data..."):
                            chunks_added = run_async(rag_system.load_analysis_results(selected_file))
                        st.success(f"✅ Loaded {chunks_added} chunks into RAG system")
                    except Exception as e:
                        st.error(f"❌ Failed to load data: {e}")
    
    # Load from current session results
    if st.session_state.analysis_results:
        if st.button("📥 Load Current Session Results"):
            try:
                from langchain_core.documents import Document
                
                documents = []
                for result in st.session_state.analysis_results:
                    if result.analysis:
                        doc = Document(
                            page_content=result.analysis,
                            metadata={
                                'source': result.url,
                                'timestamp': result.timestamp,
                                'type': 'session_analysis'
                            }
                        )
                        documents.append(doc)
                
                if documents:
                    with st.spinner("Loading session data..."):
                        chunks_added = run_async(rag_system.add_documents(documents))
                    st.success(f"✅ Loaded {chunks_added} chunks from current session")
                else:
                    st.warning("⚠️ No analysis results found in current session")
                    
            except Exception as e:
                st.error(f"❌ Failed to load session data: {e}")
    
    # RAG system stats
    try:
        stats = run_async(rag_system.get_collection_stats())
        if stats:
            st.info(f"📊 RAG System: {stats.get('total_documents', 'Unknown')} documents loaded")
    except:
        pass
    
    # Question interface
    st.subheader("💬 Ask Questions")
    
    # Conversation settings
    with st.expander("⚙️ Conversation Settings", expanded=False):
        conversation_id = st.text_input(
            "Conversation ID:",
            value=st.session_state.current_conversation_id
        )
        if conversation_id != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = conversation_id
            st.session_state.conversation_history = []
        
        if st.button("🔄 Start New Conversation"):
            st.session_state.current_conversation_id = f"conv_{int(time.time())}"
            st.session_state.conversation_history = []
            st.rerun()
    
    # Sample questions
    st.markdown("**💡 Sample Questions:**")
    sample_questions = [
        "What are the main topics covered in the videos?",
        "Can you summarize the key takeaways?",
        "What technical details were mentioned?",
        "What are the step-by-step processes shown?",
        "What tools or resources were referenced?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, question in enumerate(sample_questions):
        with cols[i % len(cols)]:
            if st.button(f"📝 {question[:30]}...", key=f"sample_{i}"):
                ask_question(rag_system, question)
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about the videos?"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🤔 Ask Question", disabled=not question):
            ask_question(rag_system, question)
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("💭 Conversation History")
        
        for i, (q, a, timestamp) in enumerate(st.session_state.conversation_history):
            with st.container():
                # Question
                st.markdown(f"**👤 Question {i+1}** ({timestamp}):")
                st.markdown(f"> {q}")
                
                # Answer
                st.markdown("**🤖 Answer:**")
                st.markdown(a)
                
                st.markdown("---")

def ask_question(rag_system, question):
    """Ask a question using the RAG system"""
    if not check_api_keys():
        return
    
    try:
        with st.spinner("🤔 Thinking..."):
            response = run_async(rag_system.ask_question(
                question,
                conversation_id=st.session_state.current_conversation_id
            ))
        
        # Add to conversation history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation_history.append((question, response.answer, timestamp))
        
        # Display response
        st.success("✅ Answer generated!")
        
        # Show sources if available
        if response.sources:
            with st.expander(f"📚 Sources ({len(response.sources)} documents)", expanded=False):
                for i, source in enumerate(response.sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(f"- URL: {source.metadata.get('source', 'Unknown')}")
                    st.markdown(f"- Content: {truncate_text(source.page_content, 200)}")
                    st.markdown("---")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to get answer: {e}")

# Analysis Results Viewer
def results_viewer():
    """Interface for viewing and managing analysis results"""
    st.header("📁 Analysis Results Manager")
    
    # Results directory
    results_dir = Path("analysis-results")
    
    if not results_dir.exists():
        st.info("📂 No analysis results directory found. Run some analyses first!")
        return
    
    # List available result files
    json_files = list(results_dir.glob("analysis-*.json"))
    
    if not json_files:
        st.info("📄 No analysis result files found. Run some analyses first!")
        return
    
    st.success(f"📊 Found {len(json_files)} analysis result files")
    
    # File selector
    selected_file = st.selectbox(
        "Select result file:",
        options=json_files,
        format_func=lambda x: f"{x.name} (Modified: {datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
    )
    
    if selected_file:
        try:
            # Load and display results
            with open(selected_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            batch_results = BatchAnalysisResults(
                timestamp=data['timestamp'],
                total_videos=data['total_videos'],
                results=[AnalysisResult(**result) for result in data['results']]
            )
            
            # Display results
            display_analysis_results(batch_results)
            
            # Download option
            st.download_button(
                label="💾 Download Results",
                data=json.dumps(data, indent=2),
                file_name=selected_file.name,
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Failed to load results: {e}")

# Settings and Configuration
def settings_interface():
    """Interface for application settings"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    nomic_api_key = os.getenv('NOMIC_API_KEY')
    
    if google_api_key:
        st.success("✅ Google API Key configured")
        st.text(f"Key: {'*' * (len(google_api_key) - 8) + google_api_key[-8:]}")
    else:
        st.error("❌ Google API Key not configured")
    
    if nomic_api_key:
        st.success("✅ Nomic API Key configured")
        st.text(f"Key: {'*' * (len(nomic_api_key) - 8) + nomic_api_key[-8:]}")
    else:
        st.warning("⚠️ Nomic API Key not configured (optional)")
    
    # System Information
    st.subheader("📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Environment:**")
        st.code(f"""
Python: {sys.version}
Working Directory: {os.getcwd()}
Project Root: {Path(__file__).parent.parent}
        """)
    
    with col2:
        st.markdown("**Directories:**")
        dirs_info = {
            "Analysis Results": "analysis-results/",
            "Chroma DB": "streamlit_chroma_db/",
            "Examples": "examples/",
            "Tests": "tests/"
        }
        
        for name, path in dirs_info.items():
            full_path = Path(path)
            if full_path.exists():
                st.success(f"✅ {name}: {path}")
            else:
                st.warning(f"⚠️ {name}: {path} (not found)")
    
    # Clear Data
    st.subheader("🗑️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.conversation_history = []
            st.success("✅ Conversation history cleared")
    
    with col2:
        if st.button("🗑️ Clear Session Results"):
            st.session_state.analysis_results = []
            st.success("✅ Session results cleared")
    
    # Reset Application
    if st.button("🔄 Reset Application", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Application reset! Please refresh the page.")

# Main Application
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 YouTube RAG Analyzer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Analyze YouTube videos with AI and ask intelligent questions about the content
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=80)
        st.title("Navigation")
        
        page = st.selectbox(
            "Choose a page:",
            [
                "🎥 Video Analysis",
                "🧠 Ask Questions",
                "📁 Results Viewer",
                "⚙️ Settings"
            ]
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("**📊 Quick Stats**")
        st.metric("Conversations", len(st.session_state.conversation_history))
        st.metric("Session Results", len(st.session_state.analysis_results))
        
        st.markdown("---")
        
        # Help
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **Getting Started:**
            1. Add your YouTube URLs
            2. Run video analysis
            3. Ask questions about the content
            
            **Tips:**
            - Use custom prompts for specific analysis
            - Load previous results for querying
            - Try sample questions for ideas
            
            **Troubleshooting:**
            - Ensure API keys are configured
            - Check URL validity
            - Review error messages
            """)
    
    # Main content based on selected page
    if page == "🎥 Video Analysis":
        analyze_videos_interface()
    elif page == "🧠 Ask Questions":
        rag_interface()
    elif page == "📁 Results Viewer":
        results_viewer()
    elif page == "⚙️ Settings":
        settings_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎬 YouTube RAG Analyzer | Built with Streamlit, LangChain, and Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()