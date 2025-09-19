# 🎬 YouTube RAG Analyzer - Quick Start Guide

Welcome to the YouTube RAG Analyzer! This guide will help you get started with the Streamlit web interface.

## 🚀 Quick Setup

1. **Navigate to the project directory**:
   ```bash
   cd py_youtube_rag
   ```

2. **Activate conda environment** (if using conda):
   ```bash
   conda activate <your venv>
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   - Copy `.env.example` to `.env`
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     ```

5. **Launch the web interface**:
   ```bash
   python launch.py
   ```
   Or directly:
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Open your browser** to: http://localhost:8501

## 🎯 Using the Web Interface

### 📹 Video Analysis Tab
1. **Choose input method**:
   - Single video: Enter one YouTube URL
   - Multiple videos: Enter multiple URLs (one per line)
   - Upload file: Upload a text file with URLs

2. **Configure analysis**:
   - Set custom prompts for specific analysis goals
   - Adjust rate limiting and concurrency
   - Click "Start Analysis"

3. **View results**:
   - Real-time progress tracking
   - Summary metrics and visualizations
   - Individual video analysis results
   - Download results as JSON

### 🧠 Ask Questions Tab
1. **Load analysis data**:
   - Select previous analysis files
   - Or load current session results

2. **Ask questions**:
   - Use sample questions or write your own
   - Get AI-powered answers based on video content
   - View conversation history
   - See source citations

3. **Manage conversations**:
   - Create new conversation threads
   - Clear history when needed
   - Export conversation data

### 📁 Results Viewer Tab
- Browse all analysis result files
- View detailed breakdowns
- Download individual results
- Compare analysis across sessions

### ⚙️ Settings Tab
- Check API key configuration
- View system information
- Manage data and cache
- Reset application state

## 💡 Tips for Best Results

### Video Analysis
- **Use specific prompts**: "Extract the step-by-step registration process" works better than "Analyze this video"
- **Batch processing**: Analyze multiple related videos together for better context
- **Rate limiting**: Adjust based on your API quota and needs

### Questions & RAG
- **Load relevant data first**: Make sure to load analysis results before asking questions
- **Be specific**: Ask detailed questions for better answers
- **Use conversation threads**: Keep related questions in the same conversation for context

### Performance
- **Monitor API usage**: Check your Google Cloud console for quota usage
- **Clear cache**: Use the settings tab to clear data if needed
- **Check memory**: Large batches may require system resources

## 🎬 Example Workflow

1. **Start with analysis**:
   ```
   URLs: 
   https://www.youtube.com/watch?v=VIDEO_ID_1
   https://www.youtube.com/watch?v=VIDEO_ID_2
   
   Custom Prompt: "Focus on technical implementation details and best practices"
   ```

2. **Load results into RAG**:
   - Go to "Ask Questions" tab
   - Click "Load Current Session Results"

3. **Ask intelligent questions**:
   ```
   "What are the key technical requirements mentioned?"
   "Can you summarize the implementation steps?"
   "What tools and technologies were recommended?"
   ```

4. **Export and save**:
   - Download analysis results
   - Save conversation history
   - Use results for further research

## 🔧 Troubleshooting

### Common Issues

**"API Key not found"**
- Check your `.env` file exists
- Verify the API key is correct
- Restart the Streamlit app

**"Import errors"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check you're in the correct conda environment

**"Connection errors"**
- Check your internet connection
- Verify API quotas aren't exceeded
- Try reducing concurrency in settings

**"Streamlit won't start"**
- Check if port 8501 is available
- Try: `streamlit run streamlit_app.py --server.port 8502`
- Restart your terminal/command prompt

### Getting Help

1. **Check the logs**: Streamlit shows detailed error messages
2. **Review examples**: Look at `examples/basic_usage.py`
3. **Test components**: Run `python setup.py` to validate setup
4. **Check documentation**: Refer to the main README.md

## 🌟 Advanced Features

### Custom Analysis Prompts
Create specialized prompts for different video types:
- **Educational**: "Extract learning objectives, key concepts, and exercises"
- **Technical**: "Focus on code examples, APIs, and implementation details"
- **Business**: "Identify strategies, metrics, and actionable insights"

### RAG Conversations
- **Follow-up questions**: Build on previous answers
- **Comparative analysis**: "How do these videos compare?"
- **Deep dives**: "Explain this concept in more detail"

### Data Management
- **Organize results**: Use descriptive conversation IDs
- **Export data**: Download for external analysis
- **Backup**: Save important conversations and results

---

**Ready to analyze? 🚀 Launch the app and start exploring your YouTube content!**

For more detailed information, see the main [README.md](README.md) file.