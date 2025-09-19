"""
Utility functions for YouTube RAG Analyzer

Common helper functions used across the project.
"""

import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse, parse_qs


def is_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube URL
    
    Args:
        url: URL to check
        
    Returns:
        True if it's a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'youtube\.com/watch\?v=',
        r'youtu\.be/',
        r'youtube\.com/embed/',
        r'youtube\.com/v/',
    ]
    
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various YouTube URL formats
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    # Standard youtube.com/watch?v= format
    if 'youtube.com/watch' in url:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get('v', [None])[0]
    
    # Short youtu.be/ format
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    
    # Embed format
    elif 'youtube.com/embed/' in url:
        return url.split('embed/')[-1].split('?')[0]
    
    # Other formats
    elif 'youtube.com/v/' in url:
        return url.split('v/')[-1].split('?')[0]
    
    return None


def normalize_youtube_url(url: str) -> Optional[str]:
    """
    Normalize YouTube URL to standard format
    
    Args:
        url: YouTube URL in any format
        
    Returns:
        Normalized URL or None if invalid
    """
    video_id = extract_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def validate_youtube_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Validate a list of YouTube URLs
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        Dictionary with validation results
    """
    valid_urls = []
    invalid_urls = []
    normalized_urls = []
    
    for url in urls:
        if is_youtube_url(url):
            normalized = normalize_youtube_url(url)
            if normalized:
                valid_urls.append(url)
                normalized_urls.append(normalized)
            else:
                invalid_urls.append(url)
        else:
            invalid_urls.append(url)
    
    return {
        'valid_urls': valid_urls,
        'invalid_urls': invalid_urls,
        'normalized_urls': normalized_urls,
        'total': len(urls),
        'valid_count': len(valid_urls),
        'invalid_count': len(invalid_urls)
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and dots
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = re.sub(r'\.+', '.', filename)
    
    return filename


def load_json_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with error handling
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a simple text progress bar
    
    Args:
        current: Current progress
        total: Total items
        width: Width of progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        percentage = 100
    else:
        percentage = (current / total) * 100
    
    filled = int(width * current // total) if total > 0 else width
    bar = '█' * filled + '░' * (width - filled)
    
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of tokens in text
    (approximately 4 characters per token for English)
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def run_with_timeout(coro, timeout: float, default=None):
    """
    Run coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout
        
    Returns:
        Result or default value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics
    
    Args:
        text: Text to analyze
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of key phrases
    """
    # Simple extraction based on capitalized words and common patterns
    import re
    
    # Find capitalized phrases (potential proper nouns, titles)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Find quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)
    
    # Combine and deduplicate
    phrases = list(set(capitalized + quoted))
    
    # Sort by length (longer phrases often more important)
    phrases.sort(key=len, reverse=True)
    
    return phrases[:max_phrases]


def format_analysis_summary(analysis: str, max_length: int = 200) -> str:
    """
    Create a summary of video analysis
    
    Args:
        analysis: Full analysis text
        max_length: Maximum summary length
        
    Returns:
        Formatted summary
    """
    # Extract first few sentences or key points
    sentences = analysis.split('. ')
    
    summary = ""
    for sentence in sentences:
        if len(summary + sentence) < max_length:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip()


class ColoredOutput:
    """Simple colored console output"""
    
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    
    @classmethod
    def print(cls, text: str, color: str = 'white', bold: bool = False):
        """Print colored text"""
        color_code = cls.COLORS.get(color, cls.COLORS['white'])
        bold_code = cls.COLORS['bold'] if bold else ''
        end_code = cls.COLORS['end']
        
        print(f"{bold_code}{color_code}{text}{end_code}")
    
    @classmethod
    def success(cls, text: str):
        """Print success message"""
        cls.print(f"✅ {text}", 'green')
    
    @classmethod
    def error(cls, text: str):
        """Print error message"""
        cls.print(f"❌ {text}", 'red')
    
    @classmethod
    def warning(cls, text: str):
        """Print warning message"""
        cls.print(f"⚠️ {text}", 'yellow')
    
    @classmethod
    def info(cls, text: str):
        """Print info message"""
        cls.print(f"ℹ️ {text}", 'blue')