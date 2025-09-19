"""
YouTube Video Analyzer - Python Implementation

This module provides YouTube video analysis capabilities using Google's Gemini AI,
equivalent to the TypeScript analyzeVideos.ts implementation.

Key features:
- Direct YouTube URL processing (no downloads required)
- Comprehensive video content analysis
- Batch processing with intelligent rate limiting
- Custom prompts and analysis templates
- Structured JSON output
- Error handling and retry logic
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import aiohttp
from google import genai
from google.genai import types
from asyncio_throttle import Throttler

# Import centralized configuration
from config import Config


@dataclass
class AnalysisResult:
    """Structure for individual video analysis results"""
    url: str
    analysis: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = ""
    duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BatchAnalysisResults:
    """Structure for batch analysis results"""
    timestamp: str
    total_videos: int
    results: List[AnalysisResult]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class YouTubeAnalyzer:
    """
    YouTube Video Analyzer using Google Gemini AI
    
    Provides functionality equivalent to the TypeScript analyzeVideos.ts:
    - Direct YouTube URL processing 
    - Batch analysis with rate limiting
    - Comprehensive content extraction
    - Structured output generation
    """
    
    DEFAULT_PROMPT = """
    Please provide a comprehensive analysis of this YouTube video. Extract and organize all the important information presented in the video including:
    
    ## 📋 Video Overview:
    1. **Video Title and Main Topic**: What is the video about?
    2. **Duration and Key Sections**: Major sections or timestamps covered
    3. **Target Audience**: Who is this video intended for?
    4. **Purpose and Objectives**: What does the video aim to achieve?
    
    ## 🎯 Content Analysis:
    1. **Key Topics and Themes**: Main subjects discussed
    2. **Step-by-Step Processes**: Any procedures, tutorials, or workflows shown
    3. **Important Information**: Facts, statistics, or crucial details mentioned
    4. **Technical Details**: Specifications, requirements, or technical information
    5. **Tips and Best Practices**: Advice or recommendations provided
    
    ## 📚 Educational Content:
    1. **Learning Objectives**: What viewers can learn
    2. **Skills or Knowledge**: What skills or knowledge are being taught
    3. **Examples and Demonstrations**: Practical examples shown
    4. **Common Mistakes**: Pitfalls or errors to avoid (if mentioned)
    
    ## 🔍 Additional Insights:
    1. **Context and Background**: Background information provided
    2. **Related Topics**: Connected subjects or references
    3. **Resources Mentioned**: Links, tools, or materials referenced
    4. **Contact Information**: Any contact details or support information
    5. **Next Steps**: Follow-up actions or recommendations
    
    ## 💡 Key Takeaways:
    1. **Main Messages**: Core messages or conclusions
    2. **Action Items**: What viewers should do after watching
    3. **Important Notes**: Critical information to remember
    
    Please provide a detailed, well-structured analysis with clear headings and bullet points. Focus on extracting actionable information and organizing it in a way that would be useful for someone who needs to reference this content later.
    
    Format your response with clear markdown headings and organized sections for easy reference and processing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        rate_limit_delay: float = None,
        max_concurrent: int = None,
        max_retries: int = 3
    ):
        """
        Initialize the YouTube Analyzer
        
        Args:
            api_key: Google API key (will use config/env var if not provided)
            model: Gemini model to use (defaults to config)
            rate_limit_delay: Delay between requests in seconds (defaults to config)
            max_concurrent: Maximum concurrent requests (defaults to config)
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key or Config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY in .env file or pass api_key parameter.\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        self.model = model or Config.DEFAULT_MODEL
        self.rate_limit_delay = rate_limit_delay or Config.DEFAULT_RATE_LIMIT_DELAY
        self.max_concurrent = max_concurrent or Config.MAX_CONCURRENT_ANALYSES
        self.max_retries = max_retries
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize throttler for rate limiting
        self.throttler = Throttler(rate_limit=1/self.rate_limit_delay)
        
        # Results directory
        self.results_dir = Path("analysis-results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def analyze_video(
        self,
        video_url: str,
        custom_prompt: Optional[str] = None,
        timeout: int = 120
    ) -> AnalysisResult:
        """
        Analyze a single YouTube video
        
        Args:
            video_url: YouTube video URL
            custom_prompt: Optional custom analysis prompt
            timeout: Request timeout in seconds
            
        Returns:
            AnalysisResult with analysis or error information
        """
        print(f"\n🎥 Analyzing video: {video_url}")
        print("⏳ Processing with Gemini...")
        
        start_time = time.time()
        prompt = custom_prompt or self.DEFAULT_PROMPT
        
        try:
            # Apply rate limiting
            async with self.throttler:
                # Create the content for Gemini API
                response = await asyncio.wait_for(
                    self._make_gemini_request(video_url, prompt),
                    timeout=timeout
                )
                
                duration = time.time() - start_time
                
                return AnalysisResult(
                    url=video_url,
                    analysis=response.text,
                    duration_seconds=duration
                )
                
        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {timeout} seconds"
            print(f"❌ {error_msg}")
            return AnalysisResult(
                url=video_url,
                error=error_msg,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as error:
            error_msg = f"Error analyzing video: {str(error)}"
            print(f"❌ {error_msg}")
            return AnalysisResult(
                url=video_url,
                error=error_msg,
                duration_seconds=time.time() - start_time
            )
    
    async def _make_gemini_request(self, video_url: str, prompt: str):
        """Make the actual Gemini API request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            types.Part(text=prompt),
                            types.Part(file_data=types.FileData(file_uri=video_url))
                        ],
                        config=types.GenerateContentConfig(
                            max_output_tokens=4000,
                            temperature=0.1
                        )
                    )
                )
                return response
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                print(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    async def analyze_videos(
        self,
        video_urls: List[str],
        custom_prompt: Optional[str] = None,
        save_results: bool = True
    ) -> BatchAnalysisResults:
        """
        Analyze multiple YouTube videos with batch processing
        
        Args:
            video_urls: List of YouTube video URLs
            custom_prompt: Optional custom analysis prompt
            save_results: Whether to save results to JSON file
            
        Returns:
            BatchAnalysisResults with all analysis results
        """
        print("🚀 Starting YouTube Video Analysis")
        print(f"📹 Processing {len(video_urls)} video(s)")
        print(f"⚙️ Max concurrent: {self.max_concurrent}")
        print(f"⏱️ Rate limit: {self.rate_limit_delay}s between requests")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def analyze_with_semaphore(url: str) -> AnalysisResult:
            async with semaphore:
                return await self.analyze_video(url, custom_prompt)
        
        # Process videos concurrently
        tasks = [analyze_with_semaphore(url) for url in video_urls]
        results = []
        
        # Process with progress reporting
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            
            print(f"\n📹 Completed video {i}/{len(video_urls)}")
            if result.analysis:
                print("✅ Analysis successful")
                print(f"⏱️ Duration: {result.duration_seconds:.2f}s")
            else:
                print(f"❌ Analysis failed: {result.error}")
        
        # Create batch results
        batch_results = BatchAnalysisResults(
            timestamp=datetime.now().isoformat(),
            total_videos=len(video_urls),
            results=results
        )
        
        # Save results if requested
        if save_results:
            await self.save_results(batch_results)
        
        # Print summary
        successful = len([r for r in results if r.analysis])
        failed = len(results) - successful
        
        print(f"\n📊 ANALYSIS COMPLETE")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📄 Total videos processed: {len(results)}")
        
        return batch_results
    
    async def save_results(self, results: BatchAnalysisResults) -> Path:
        """
        Save analysis results to JSON file
        
        Args:
            results: BatchAnalysisResults to save
            
        Returns:
            Path to saved file
        """
        try:
            # Create filename with timestamp
            timestamp_str = results.timestamp.replace(":", "-").replace(".", "-")
            filename = f"analysis-{timestamp_str}.json"
            filepath = self.results_dir / filename
            
            # Convert to dict and save
            results_dict = asdict(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {filepath}")
            return filepath
            
        except Exception as error:
            print(f"❌ Error saving results: {error}")
            raise
    
    async def load_results(self, filepath: Union[str, Path]) -> BatchAnalysisResults:
        """
        Load analysis results from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            BatchAnalysisResults loaded from file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to dataclasses
            results = [AnalysisResult(**result) for result in data['results']]
            
            return BatchAnalysisResults(
                timestamp=data['timestamp'],
                total_videos=data['total_videos'],
                results=results
            )
            
        except Exception as error:
            print(f"❌ Error loading results: {error}")
            raise


async def main():
    """Main execution function for testing"""
    try:
        print('🔑 Checking API key...')
        api_key = os.getenv("GOOGLE_API_KEY")
        print('🔑 API Key loaded:', 'Yes ✅' if api_key else 'No ❌')
        
        if not api_key:
            raise ValueError('GOOGLE_API_KEY not found in environment variables')
        
        # Initialize analyzer
        analyzer = YouTubeAnalyzer()
        
        # Test with a single video
        print('\n🧪 Testing with a sample video...')
        test_result = await analyzer.analyze_video(
            'https://www.youtube.com/watch?v=WsEQjeZoEng',
            'Provide a comprehensive summary of the key points and main topics covered in this video.'
        )
        
        print('\n📊 TEST RESULTS:')
        print('=' * 80)
        if test_result.analysis:
            print(test_result.analysis)
        else:
            print(f"Error: {test_result.error}")
        print('=' * 80)
        
    except Exception as error:
        print(f'💥 Fatal error: {error}')
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    exit(exit_code)