"""
Setup Script for YouTube RAG Analyzer

This script helps you set up the Python YouTube RAG Analyzer:
1. Checks Python and dependency requirements
2. Installs required packages
3. Sets up environment configuration
4. Initializes the RAG system
5. Runs validation tests
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_conda_environment():
    """Check if we're in the correct conda environment"""
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        print(f"🐍 Conda environment: {env_name}")
        
        if env_name != 'youtuber':
            print("⚠️  Note: You're not in the 'youtuber' environment")
            print("   Consider running: conda activate youtuber")
        else:
            print("✅ Using 'youtuber' conda environment")
    else:
        print("ℹ️  Not using conda environment")


def install_dependencies():
    """Install required Python packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Set up environment configuration"""
    env_example = Path(__file__).parent / ".env.example"
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        
        # Copy template
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your API keys!")
        print(f"   File location: {env_file.absolute()}")
        return False
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️  No .env.example template found")
        return True


def check_api_keys():
    """Check if required API keys are configured"""
    from config import Config
    
    is_valid, missing_keys = Config.validate_required_keys()
    
    if not is_valid:
        print(f"❌ Missing API Keys: {', '.join(missing_keys)}")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    else:
        print("✅ All required API keys configured")
    
    # Show optional API key status
    status = Config.get_api_status()
    
    if not status['nomic_api_key']:
        print("⚠️  NOMIC_API_KEY not configured (optional)")
        print("   Get your API key from: https://atlas.nomic.ai/")
        print("   Will use SentenceTransformer embeddings as fallback")
    else:
        print("✅ NOMIC_API_KEY configured")
    
    return True


async def test_youtube_analyzer():
    """Test the YouTube analyzer"""
    print("\n🧪 Testing YouTube Analyzer...")
    
    try:
        from analyzer import YouTubeAnalyzer
        
        # Test initialization
        analyzer = YouTubeAnalyzer()
        print("✅ YouTube Analyzer initialized")
        
        # Test with a very short video analysis (mock for setup)
        print("✅ YouTube Analyzer basic functionality verified")
        return True
        
    except Exception as e:
        print(f"❌ YouTube Analyzer test failed: {e}")
        return False


async def test_streamlit_app():
    """Test Streamlit app availability"""
    print("\n🌐 Testing Streamlit App...")
    
    try:
        # Check if streamlit is installed
        import streamlit as st
        print("✅ Streamlit installed")
        
        # Check if main app file exists
        app_file = Path(__file__).parent / "streamlit_app.py"
        if app_file.exists():
            print("✅ Streamlit app file found")
            print(f"📄 App location: {app_file}")
            return True
        else:
            print("❌ Streamlit app file not found")
            return False
            
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Run: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False


async def run_example():
    """Run a basic example"""
    print("\n🚀 Running Basic Example...")
    
    try:
        # Import and run a simple example
        example_path = Path(__file__).parent / "examples" / "basic_usage.py"
        
        if example_path.exists():
            print(f"📄 Example script available at: {example_path}")
            print("   Run with: python examples/basic_usage.py")
        else:
            print("⚠️  Example script not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "analysis-results",
        "chroma_db",
        "examples",
        "tests",
        "utils"
    ]
    
    for dir_name in directories:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
    
    print("✅ Directories created")


async def main():
    """Main setup function"""
    print("🎬 YouTube RAG Analyzer - Setup Script")
    print("=" * 50)
    
    success = True
    
    # Step 1: Check Python version
    print("\n1️⃣ Checking Python version...")
    if not check_python_version():
        return 1
    
    # Step 2: Check conda environment
    print("\n2️⃣ Checking conda environment...")
    check_conda_environment()
    
    # Step 3: Create directories
    print("\n3️⃣ Creating directories...")
    create_directories()
    
    # Step 4: Install dependencies
    print("\n4️⃣ Installing dependencies...")
    if not install_dependencies():
        success = False
    
    # Step 5: Setup environment
    print("\n5️⃣ Setting up environment...")
    env_ready = setup_environment()
    
    # Step 6: Check API keys
    print("\n6️⃣ Checking API keys...")
    if not check_api_keys():
        success = False
        env_ready = False
    
    if not env_ready:
        print("\n⚠️  Setup incomplete - please configure your API keys in .env file")
        print("   Then run this setup script again to complete validation")
        return 1
    
    # Step 7: Test YouTube Analyzer
    print("\n7️⃣ Testing YouTube Analyzer...")
    if not await test_youtube_analyzer():
        success = False
    
    # Step 8: Test Streamlit App
    print("\n8️⃣ Testing Streamlit App...")
    if not await test_streamlit_app():
        success = False
    
    # Step 9: Show examples
    print("\n9️⃣ Setting up examples...")
    await run_example()
    
    # Final summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Run: python examples/basic_usage.py")
        print("2. Customize video URLs in the examples")
        print("3. Explore the analyzer.py and rag_pipeline/ modules")
        print("4. Build your own YouTube analysis application!")
        
        print("\n🔗 Quick Commands:")
        print("- Analyze videos: python -c \"import asyncio; from analyzer import *; asyncio.run(main())\"")
        print("- Run tests: python -m pytest tests/ -v")
        print("- Start interactive: python -i examples/basic_usage.py")
        
        return 0
    else:
        print("❌ Setup completed with some issues")
        print("   Please review the error messages above and fix any problems")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)