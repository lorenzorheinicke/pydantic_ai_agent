import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import logfire
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables from .env file
load_dotenv()

console = Console()

class VideoInfo(BaseModel):
    title: str
    description: Optional[str]
    transcript: str
    video_id: str
    language: str
    available_languages: List[str]
    stats: Optional[Dict] = None

class VideoStats(BaseModel):
    """Structured video statistics from YouTube Data API."""
    title: str
    channelTitle: str
    publishedAt: str
    viewCount: str
    likeCount: str
    commentCount: str
    thumbnails: Dict[str, Dict[str, Union[str, int]]]

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    raise ValueError("Invalid YouTube URL")

def fetch_transcript(video_id: str, lang: Optional[str] = None) -> dict:
    """Internal function to fetch transcript from Supadata.ai API."""
    api_key = os.getenv("SUPADATA_API_KEY")
    if not api_key:
        logfire.error("SUPADATA_API_KEY not found in environment variables")
        raise ValueError("SUPADATA_API_KEY not found in environment variables")

    params = {
        "videoId": video_id,
        "text": True
    }
    if lang:
        params["lang"] = lang

    try:
        logfire.info("Making request to Supadata API", video_id=video_id, params=params)
        response = requests.get(
            "https://api.supadata.ai/v1/youtube/transcript",
            params=params,
            headers={"x-api-key": api_key}
        )

        if response.status_code == 429:
            logfire.error("Rate limit exceeded", video_id=video_id, status_code=response.status_code)
            raise Exception("Rate limit exceeded. Please wait a few minutes before trying again.")
        elif response.status_code == 404:
            logfire.error("Transcript not available", video_id=video_id, status_code=response.status_code)
            raise Exception("Transcript not available for this video.")
        elif response.status_code != 200:
            error_msg = response.json().get('error', 'Unknown error occurred')
            logfire.error("Failed to fetch transcript", 
                video_id=video_id, 
                status_code=response.status_code,
                error=error_msg
            )
            raise Exception(f"Failed to fetch transcript: {error_msg}")

        logfire.info("Successfully fetched transcript", video_id=video_id)
        return response.json()
    except requests.exceptions.RequestException as e:
        logfire.error("Network error while fetching transcript", 
            video_id=video_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise Exception(f"Network error while fetching transcript: {str(e)}")

def process_transcript_data(transcript_data: Union[List[Dict], Dict]) -> tuple[str, str, List[str]]:
    """Process transcript data from the API into required format."""
    if isinstance(transcript_data, list):
        # If it's a list of segments, join them
        segments = []
        for segment in transcript_data:
            if isinstance(segment, dict) and 'text' in segment:
                segments.append(segment['text'])
        content = " ".join(segments)
        language = transcript_data[0].get('lang', 'en') if transcript_data else 'en'
        available_langs = [language]
    elif isinstance(transcript_data, dict):
        if 'content' in transcript_data:
            # If it's a dictionary with content field
            content = str(transcript_data['content'])
            language = transcript_data.get('lang', 'en')
            available_langs = transcript_data.get('availableLangs', [language])
        else:
            # Handle case where transcript might be nested
            segments = []
            for item in transcript_data.get('transcript', []):
                if isinstance(item, dict) and 'text' in item:
                    segments.append(item['text'])
            content = " ".join(segments) if segments else ""
            language = transcript_data.get('lang', 'en')
            available_langs = transcript_data.get('availableLangs', [language])
    else:
        content = ""
        language = "en"
        available_langs = ["en"]
    
    return content, language, available_langs

def truncate_transcript(transcript: str, max_chars: int = 100000) -> str:
    """Truncate transcript to a reasonable length while preserving meaning.
    
    Args:
        transcript: The full transcript text
        max_chars: Maximum number of characters to keep
    
    Returns:
        Truncated transcript with indication if it was truncated
    """
    if len(transcript) <= max_chars:
        return transcript
    
    # Take first 45% and last 45% of allowed length to keep both start and end context
    first_part_len = int(max_chars * 0.45)
    last_part_len = int(max_chars * 0.45)
    
    first_part = transcript[:first_part_len]
    last_part = transcript[-last_part_len:]
    
    return f"{first_part}\n\n[...transcript truncated for length...]\n\n{last_part}"

def get_youtube_video_info(video_id: str) -> VideoStats:
    """Fetch video information from YouTube Data API.
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        VideoStats object containing video information
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables")

    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,statistics"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if "items" in data and len(data["items"]) > 0:
                video_info = data["items"][0]
                snippet = video_info["snippet"]
                statistics = video_info["statistics"]
                
                # Create a stats dictionary with consistent keys
                stats_dict = {
                    "title": snippet.get("title", ""),
                    "channelTitle": snippet.get("channelTitle", ""),
                    "publishedAt": snippet.get("publishedAt", ""),
                    "viewCount": statistics.get("viewCount", "0"),
                    "likeCount": statistics.get("likeCount", "0"),
                    "commentCount": statistics.get("commentCount", "0"),
                    "thumbnails": snippet.get("thumbnails", {})
                }
                
                return VideoStats(**stats_dict)
            else:
                raise Exception("No video found with this ID")
        else:
            raise Exception(f"YouTube API error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error while fetching video info: {str(e)}")

def fetch_video_stats(video_id: str) -> VideoStats:
    """Fetch video statistics from YouTube Data API."""
    try:
        video_stats = get_youtube_video_info(video_id)
        return video_stats
    except Exception as e:
        raise Exception(f"Failed to fetch video stats: {str(e)}")

# Initialize AI agent with research-focused system prompt
agent = Agent(
    'claude-3-5-sonnet-latest',
    system_prompt="""You are a YouTube video research assistant. Your task is to:
1. Analyze video content through transcripts
2. Provide detailed summaries and insights
3. Answer questions about the video content
4. Extract key points and learning outcomes
5. Identify main themes and topics

When answering follow-up questions, use the provided video summary and transcript summary.
If you need more details, mention that they are available in the full transcript file.

Format your responses in markdown with appropriate sections and formatting."""
)

@agent.tool
def read_transcript_file(ctx: RunContext[None], file_path: str) -> str:
    """Read the full transcript from a file when needed."""
    try:
        return Path(file_path).read_text()
    except Exception as e:
        return f"Error reading transcript: {str(e)}"

def get_video_info(url: str, lang: Optional[str] = None, fetch_stats: bool = False, should_fetch_transcript: bool = True) -> VideoInfo:
    """Get video information including transcript using Supadata.ai API."""
    try:
        logfire.info("Starting video info fetch", url=url, fetch_stats=fetch_stats, fetch_transcript=should_fetch_transcript)
        video_id = extract_video_id(url)
        logfire.info("Extracted video ID", video_id=video_id)
        
        # Initialize variables
        content = ""
        language = "en"
        available_langs = ["en"]
        stats = None
        
        # Try to fetch transcript if requested
        if should_fetch_transcript:
            try:
                logfire.info("Fetching transcript", video_id=video_id)
                transcript_data = fetch_transcript(video_id, lang)
                content, language, available_langs = process_transcript_data(transcript_data)
                logfire.info("Successfully processed transcript", 
                    video_id=video_id,
                    language=language,
                    transcript_length=len(content)
                )
            except Exception as e:
                logfire.error("Failed to fetch/process transcript", 
                    video_id=video_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise Exception(f"Transcript error: {str(e)}")
        
        # Try to fetch stats if requested
        if fetch_stats:
            try:
                logfire.info("Fetching video stats", video_id=video_id)
                video_stats = fetch_video_stats(video_id)
                # Convert VideoStats to dictionary
                stats = video_stats.model_dump()
                logfire.info("Successfully fetched video stats", 
                    video_id=video_id,
                    title=video_stats.title
                )
            except Exception as e:
                logfire.error("Failed to fetch video stats", 
                    video_id=video_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Don't fail the whole request if just stats fail
                console.print(f"[yellow]Warning: Could not fetch video stats: {str(e)}[/yellow]")
        
        video_info = VideoInfo(
            title=f"Video ID: {video_id}",
            description="",
            transcript=content,
            video_id=video_id,
            language=language,
            available_languages=available_langs,
            stats=stats
        )
        logfire.info("Successfully created VideoInfo object", 
            video_id=video_id,
            has_transcript=len(content) > 0,
            has_stats=stats is not None
        )
        return video_info
    except Exception as e:
        logfire.error("Failed to get video info",
            url=url,
            error=str(e),
            error_type=type(e).__name__
        )
        raise

def save_transcript_to_temp(transcript: str) -> Path:
    """Save transcript to a temporary file and return the path."""
    temp_dir = Path(tempfile.gettempdir())
    transcript_file = temp_dir / "youtube_transcript.txt"
    transcript_file.write_text(transcript)
    return transcript_file

def get_transcript_summary(transcript: str) -> str:
    """Get a brief summary of the transcript for context."""
    # Take first and last 1000 characters as a preview
    preview_length = 1000
    if len(transcript) <= preview_length * 2:
        return transcript
    
    first_part = transcript[:preview_length]
    last_part = transcript[-preview_length:]
    return f"{first_part}\n\n[... transcript continues in file ...]\n\n{last_part}"

def format_video_info(video_info: VideoInfo, include_full_transcript: bool = False) -> str:
    """Format video information for the AI agent."""
    if include_full_transcript:
        transcript_text = video_info.transcript
    else:
        transcript_text = get_transcript_summary(video_info.transcript)
    
    return f"""
Video ID: {video_info.video_id}
Title: {video_info.title}
Language: {video_info.language}
Available Languages: {', '.join(video_info.available_languages)}

Transcript:
{transcript_text}
"""

if __name__ == "__main__":
    console.print(Markdown("# YouTube Research Assistant\nEnter a YouTube URL to analyze, or type 'quit' to exit."))

    video_info = None
    transcript_file = None
    last_summary = None

    while True:
        user_input = input("> ")
        
        if user_input.lower() in ['quit', 'exit']:
            # Cleanup temp file if it exists
            if transcript_file and transcript_file.exists():
                transcript_file.unlink()
            console.print(Markdown("**Goodbye!**"))
            break
            
        if user_input.startswith(('http://', 'https://', 'www.', 'youtube.com', 'youtu.be')):
            try:
                video_info = get_video_info(user_input)
                # Save transcript to temp file
                transcript_file = save_transcript_to_temp(video_info.transcript)
                console.print(Markdown("Video loaded successfully! You can now ask questions about it."))
                
                # Initial analysis with full transcript
                result = agent.run_sync(
                    f"Analyze this video and provide a comprehensive summary:\n{format_video_info(video_info, include_full_transcript=True)}"
                )
                last_summary = result.data
                console.print(Markdown(last_summary))
                
            except Exception as e:
                console.print(f"[red]Failed to process video: {str(e)}[/red]")
                continue
        else:
            if video_info is None:
                console.print(Markdown("*Please provide a YouTube URL first.*"))
                continue
            
            # For follow-up questions, use the summary and a preview of the transcript
            prompt = f"""Previous summary of the video:
{last_summary}

Video context:
{format_video_info(video_info, include_full_transcript=False)}

If you need more specific details, you can use the read_transcript_file tool to read the full transcript from: {transcript_file}

Question: {user_input}"""

            result = agent.run_sync(prompt)
            last_summary = result.data
            console.print(Markdown(result.data)) 