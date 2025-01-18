import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        raise ValueError("SUPADATA_API_KEY not found in environment variables")

    params = {
        "videoId": video_id,
        "text": True
    }
    if lang:
        params["lang"] = lang

    response = requests.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        params=params,
        headers={"x-api-key": api_key}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to fetch transcript: {response.text}")

    return response.json()

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

def get_video_info(url: str, lang: Optional[str] = None) -> VideoInfo:
    """Get video information including transcript using Supadata.ai API."""
    try:
        video_id = extract_video_id(url)
        transcript_data = fetch_transcript(video_id, lang)
        content, language, available_langs = process_transcript_data(transcript_data)
        
        return VideoInfo(
            title=f"Video ID: {video_id}",  # Basic title with video ID
            description="",  # Empty description as we don't have this data
            transcript=content,
            video_id=video_id,
            language=language,
            available_languages=available_langs
        )
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
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