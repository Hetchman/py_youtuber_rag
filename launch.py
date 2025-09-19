"""
Launch Script for YouTube RAG Analyzer Streamlit App

Simple script to launch the Streamlit web interface with proper configuration.
Handles environment setup and provides helpful startup information.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  GOOGLE_API_KEY not found in environment")
        print("   Please set your API key in .env file")
    else:
        print("✅ Google API key configured")
    
    # Check required directories
    dirs_to_check = ['analysis-results', 'examples', 'rag_pipeline', 'utils']
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory: {dir_name}/")
        else:
            print(f"⚠️  Directory missing: {dir_name}/")
    
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching YouTube RAG Analyzer...")
    print("=" * 50)
    
    # Streamlit configuration
    streamlit_config = {
        '--server.port': '8501',
        '--server.address': 'localhost',
        '--server.headless': 'false',
        '--browser.gatherUsageStats': 'false',
        '--theme.primaryColor': '#4ECDC4',
        '--theme.backgroundColor': '#FFFFFF',
        '--theme.secondaryBackgroundColor': '#F0F2F6',
        '--theme.textColor': '#262730'
    }
    
    # Build command
    cmd = ['streamlit', 'run', 'streamlit_app.py']
    for key, value in streamlit_config.items():
        cmd.extend([key, value])
    
    try:
        print("🌐 Starting Streamlit server...")
        print(f"📍 URL: http://localhost:8501")
        print("\n💡 Tips:")
        print("   - Press Ctrl+C to stop the server")
        print("   - The app will open in your default browser")
        print("   - Refresh the page if you encounter issues")
        print("\n" + "=" * 50)
        
        # Launch Streamlit
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Install Streamlit: pip install streamlit")
        print("   2. Check if port 8501 is available")
        print("   3. Run: streamlit run streamlit_app.py")
        return False
    
    except KeyboardInterrupt:
        print("\n⚠️  Server stopped by user")
        return True
    
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        return False
    
    return True

def main():
    """Main function"""
    print("🎬 YouTube RAG Analyzer - Launch Script")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed")
        print("   Please run setup.py first: python setup.py")
        return 1
    
    # Launch application
    success = launch_streamlit()
    
    if success:
        print("\n✅ Application launched successfully!")
        return 0
    else:
        print("\n❌ Failed to launch application")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)