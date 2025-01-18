import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path

import logfire
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (ModelRequest, ModelResponse,
                                  SystemPromptPart, TextPart, UserPromptPart)

# Import only the necessary functions from youtube_researcher
from youtube_researcher import (VideoInfo, VideoStats, format_video_info,
                                get_video_info, save_transcript_to_temp)

# Configure logfire for debugging
logfire.configure()

# Apply nest_asyncio to handle event loop conflicts
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize Streamlit page
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")
st.title("AI Chat Assistant")

# Initialize the agent
@st.cache_resource
def get_agent():
    agent = Agent(
        'openai:gpt-4o-mini',
        system_prompt='''Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.

You have access to YouTube research capabilities through specific tools. Choose the appropriate tool based on what the user asks for:

1. Use analyze_youtube_video tool when the user asks for:
   - Video summaries or analysis
   - Main ideas or key points
   - Insights or themes
   - Understanding what the video is about
   - Questions about the video content

2. Use get_youtube_transcript tool when the user:
   - Specifically asks for the transcript
   - Wants the raw text/subtitles
   - Needs the exact words from the video

3. Use get_youtube_stats tool when the user asks for:
   - View count, likes, or comments
   - Channel information
   - Publication date
   - Thumbnails
   - Any numerical statistics

Do not mix tools unless specifically asked. Choose the most appropriate tool based on the user's request.''',
    )
    
    @agent.tool_plain
    def get_datetime(timezone: str = "utc") -> str:
        """Get the current date and time in the specified timezone.
        
        Args:
            timezone: Either "utc" or "local" (default: "utc")
        
        Returns:
            A string containing the formatted date and time
        """
        from datetime import UTC
        
        now = datetime.now()
        utc_now = datetime.now(UTC)
        
        if timezone.lower() == "local":
            return f"Local time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        else:
            return f"UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    
    @agent.tool
    def analyze_youtube_video(ctx: RunContext[None], url: str) -> str:
        """Analyze a YouTube video and provide insights.
        
        Args:
            url: The YouTube video URL to analyze
        
        Returns:
            A detailed analysis of the video content
        """
        try:
            # First try to get video stats
            stats = None
            try:
                video_info = get_video_info(url, fetch_stats=True, should_fetch_transcript=False)
                if video_info.stats:
                    stats = VideoStats(**video_info.stats)
            except Exception as stats_error:
                logfire.error("Failed to fetch video stats", 
                    error=str(stats_error),
                    url=url,
                    error_type=type(stats_error).__name__
                )

            # Then try to get transcript and analyze
            try:
                logfire.info("Fetching video transcript", url=url)
                video_info = get_video_info(url, should_fetch_transcript=True)
                
                # Save transcript to temp file for potential follow-up questions
                temp_dir = Path(tempfile.gettempdir())
                transcript_file = temp_dir / "youtube_transcript.txt"
                transcript_file.write_text(video_info.transcript)
                
                # Format video information for analysis
                formatted_info = format_video_info(video_info, include_full_transcript=True)
                
                # Create a YouTube research agent for analysis
                youtube_agent = Agent(
                    'claude-3-5-sonnet-latest',
                    system_prompt="""You are a YouTube video research assistant. Analyze the video content and provide:
1. A concise summary
2. Key points and insights
3. Main themes and topics
Format your response in markdown."""
                )
                
                # Get analysis from the YouTube agent
                result = youtube_agent.run_sync(
                    f"Analyze this video and provide a comprehensive but concise summary:\n{formatted_info}"
                )
                
                return result.data
                
            except Exception as transcript_error:
                logfire.error("Failed to fetch or analyze transcript",
                    error=str(transcript_error),
                    url=url,
                    error_type=type(transcript_error).__name__,
                    has_stats=stats is not None
                )
                
                # If we have stats, return those with an error message about transcript
                if stats:
                    return f"""## Video Information

Unfortunately, I couldn't access the transcript for this video at the moment. However, I can tell you about the video:

**Title**: {stats.title}
**Channel**: {stats.channelTitle}
**Published**: {stats.publishedAt}
**Stats**:
- 👀 {stats.viewCount} views
- 👍 {stats.likeCount} likes
- 💬 {stats.commentCount} comments

The transcript is currently unavailable ({str(transcript_error)}). You might want to:
1. Try again in a few minutes
2. Check if the video has closed captions
3. Try a different video"""
                else:
                    raise Exception(f"Could not analyze video: {str(transcript_error)}")
                
        except Exception as e:
            logfire.error("Failed to analyze video",
                error=str(e),
                url=url,
                error_type=type(e).__name__
            )
            return f"Error analyzing video: {str(e)}"
    
    @agent.tool
    def get_youtube_transcript(ctx: RunContext[None], url: str) -> str:
        """Get only the transcript of a YouTube video without analysis.
        
        Args:
            url: The YouTube video URL
        
        Returns:
            The raw transcript text
        """
        try:
            logfire.info("Starting transcript fetch", url=url)
            video_info = get_video_info(url, fetch_stats=False, should_fetch_transcript=True)
            logfire.info("Successfully fetched transcript", 
                url=url,
                transcript_length=len(video_info.transcript),
                language=video_info.language
            )
            return f"## Video Transcript\n\n{video_info.transcript}"
        except Exception as e:
            logfire.error("Failed to get transcript",
                error=str(e),
                url=url,
                error_type=type(e).__name__
            )
            return f"Error getting transcript: {str(e)}"
    
    @agent.tool
    def get_youtube_stats(ctx: RunContext[None], url: str) -> VideoStats:
        """Get statistics for a YouTube video without analysis.
        
        Args:
            url: The YouTube video URL
        
        Returns:
            VideoStats object containing video statistics
        """
        try:
            logfire.info("Fetching video stats", url=url)
            video_info = get_video_info(url, fetch_stats=True, should_fetch_transcript=False)
            if not video_info.stats:
                logfire.error("No stats available", url=url)
                raise Exception("No statistics available for this video.")
            
            stats = VideoStats(
                title=video_info.stats["title"],
                channelTitle=video_info.stats["channelTitle"],
                publishedAt=video_info.stats["publishedAt"],
                viewCount=video_info.stats["viewCount"],
                likeCount=video_info.stats["likeCount"],
                commentCount=video_info.stats["commentCount"],
                thumbnails=video_info.stats["thumbnails"]
            )
            logfire.info("Successfully fetched video stats", 
                url=url,
                title=stats.title,
                channel=stats.channelTitle
            )
            return stats
        except Exception as e:
            logfire.error("Failed to get video stats",
                error=str(e),
                url=url,
                error_type=type(e).__name__
            )
            raise Exception(f"Error getting video stats: {str(e)}")
    
    return agent

agent = get_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a default welcome message
    welcome_msg = asyncio.run(agent.run("Say hi and briefly explain what you can do"))
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg.data})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get agent response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert message history to proper format
        history = []
        
        # Add system prompt first
        history.append(ModelRequest(parts=[
            SystemPromptPart(content='Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.')
        ]))
        
        # Add conversation history
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
            elif msg["role"] == "assistant":
                history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
        
        # Create and run the async streaming response
        async def get_streaming_response():
            async with agent.run_stream(prompt, message_history=history) as result:
                async for chunk in result.stream_text(delta=True):
                    yield chunk

        # Process the stream
        async def process_stream():
            response_text = ""
            async for chunk in get_streaming_response():
                response_text += chunk
                message_placeholder.markdown(response_text + "▌")
            message_placeholder.markdown(response_text)
            return response_text

        # Run the async process
        full_response = asyncio.run(process_stream())
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response}) 