# YouTube RAG Analyzer - Python Implementation

A powerful Python application that uses Google's Gemini AI to analyze YouTube videos and provides a complete RAG (Retrieval-Augmented Generation) pipeline for intelligent question-answering about video content.

## 🚀 Features

- **AI-Powered YouTube Analysis**: Direct YouTube URL processing using Google Gemini 2.5 Flash
- **Modern RAG Pipeline**: LangChain + LangGraph + Nomic + Chroma integration
- **Batch Processing**: Analyze multiple videos with intelligent rate limiting
- **Conversational Interface**: Chat-style interaction with video content
- **Vector Search**: Semantic search across analyzed video content
- **State Management**: Persistent conversation history and context

## 🛠 Technology Stack

- **Google Gemini 2.5 Flash**: Video analysis and LLM responses
- **LangChain**: RAG framework and integrations
- **LangGraph**: Workflow orchestration and state management
- **Nomic Embeddings**: High-quality text embeddings
- **Chroma**: Vector database for semantic search
- **Python 3.8+**: Core implementation

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Nomic API key (optional, for embeddings)
- Conda environment 'youtuber' (recommended)

## 🔧 Installation

1. **Activate your conda environment**:
   ```bash
   conda activate <your venv>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - A `.env` file has been created in the project root
   - Edit the `.env` file and add your API keys:
   ```bash
   # Edit .env file
   GOOGLE_API_KEY=your_actual_gemini_api_key_here
   NOMIC_API_KEY=your_nomic_api_key_here  # Optional
   ```

4. **Test your setup**:
   ```bash
   python test_config.py
   ```

## 🎬 Usage

### YouTube Video Analysis

1. **Single Video Analysis**:
   ```python
   from analyzer import YouTubeAnalyzer
   
   analyzer = YouTubeAnalyzer()
   result = await analyzer.analyze_video("https://www.youtube.com/watch?v=VIDEO_ID")
   print(result.analysis)
   ```

2. **Batch Analysis**:
   ```python
   videos = [
       "https://www.youtube.com/watch?v=VIDEO_ID_1",
       "https://www.youtube.com/watch?v=VIDEO_ID_2"
   ]
   results = await analyzer.analyze_videos(videos)
   ```

### Web Interface

1. **Launch Streamlit App**:
   ```bash
   python launch.py
   ```
   Or directly:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the web interface**:
   - Open your browser to: http://localhost:8501
   - Use the intuitive web interface for analysis and querying
   - Features include:
     - Video analysis with custom prompts
     - Batch processing interface
     - RAG-powered question answering
     - Conversation history
     - Results visualization

## 📁 Project Structure

```
py_youtube_rag/
├── streamlit_app.py         # Streamlit web interface
├── launch.py               # Web app launcher script
├── .streamlit/             # Streamlit configuration
│   └── config.toml        # App theme and settings
├── rag_pipeline/
│   ├── __init__.py
│   ├── core.py             # Main RAG system
│   ├── embeddings.py       # Nomic embeddings
│   ├── vector_store.py     # Chroma integration
│   └── chat.py             # Conversational interface
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Utility functions
├── tests/
│   └── test_analyzer.py    # Test suite
├── examples/
│   └── basic_usage.py      # Usage examples
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
└── README.md             # This file
```

## 🔑 Environment Variables

The `.env` file in the project root contains all configuration settings:

### Required
```env
# Google Gemini API Key (Required)
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```
Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

### Optional
```env
# Nomic API Key for enhanced embeddings (Optional)
NOMIC_API_KEY=your_nomic_api_key_here

# LangSmith for debugging (Optional)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true

# Application Settings (defaults provided)
MAX_CONCURRENT_ANALYSES=3
DEFAULT_RATE_LIMIT_DELAY=2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_MODEL=gemini-2.5-flash
```

### Configuration Management

The system uses a centralized configuration system that:
- Loads environment variables from `.env` file automatically
- Provides default values for optional settings
- Validates API keys and shows helpful error messages
- Can be tested with: `python test_config.py`

## 📊 Key Differences from TypeScript Version

### Advantages of Python Implementation

1. **Enhanced RAG Integration**: Native LangChain ecosystem
2. **Better Vector Operations**: Optimized NumPy/SciPy operations
3. **Rich ML Ecosystem**: Integration with ML/AI libraries
4. **Improved Data Processing**: Pandas for analysis result manipulation
5. **Modern Async Support**: Full async/await pattern support

### Equivalent Features

- ✅ Direct YouTube URL processing (no downloads)
- ✅ Comprehensive video content analysis
- ✅ Batch processing with rate limiting
- ✅ Custom prompts and analysis templates
- ✅ JSON output for structured results
- ✅ Error handling and retry logic

## 🚦 Rate Limiting

The analyzer includes intelligent rate limiting:
- 2-second delay between video analyses
- Exponential backoff on API errors
- Quota management and monitoring

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## 📚 Examples

See the `examples/` directory for:
- Basic video analysis
- Batch processing
- RAG query examples
- Custom prompt usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `.env` file exists with correct keys
2. **Import Errors**: Verify all dependencies are installed
3. **Rate Limiting**: The analyzer includes automatic retry logic
4. **Memory Issues**: Batch processing includes memory management

### Getting Help

- Check the examples in `examples/` directory
- Review test cases in `tests/` for usage patterns
- Ensure conda environment 'youtuber' is activated

---

**Happy analyzing! 🎬✨**