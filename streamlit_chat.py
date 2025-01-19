import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Initialize Streamlit page configuration first
import streamlit as st

st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖")

import logfire
import nest_asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (ModelRequest, ModelResponse,
                                  SystemPromptPart, TextPart, UserPromptPart)

# Import authentication module
from auth import check_auth, logout
# Import neighborhood agent and dependencies
from neighborhood_agent import (NeighborhoodDeps, NeighborhoodInfo,
                                NoInformation, neighborhood_agent)
# Import only the necessary functions from youtube_researcher
from youtube_researcher import (VideoInfo, VideoStats, format_video_info,
                                get_video_info, save_transcript_to_temp)

# Configure logfire for debugging
logfire.configure()

# Apply nest_asyncio to handle event loop conflicts
nest_asyncio.apply()

# Create or get event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load environment variables
load_dotenv()

# Check authentication before proceeding
user_info = check_auth()

if user_info:
    # Show user info and logout button in sidebar
    with st.sidebar:
        st.image(user_info['picture'], width=50)
        st.write(f"Welcome, {user_info['name']}!")
        if st.button("Logout"):
            logout()

    st.title("AI Chat Assistant")

    # Initialize the agent
    @st.cache_resource
    def get_agent():
        """Get a configured instance of the AI Chat Agent."""
        agent = Agent(
            'openai:gpt-4o-mini',
            system_prompt='''Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.

You have access to YouTube research capabilities and neighborhood information through specific tools. Choose the appropriate tool based on what the user asks for:

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

4. Use get_neighborhood_info tool when the user asks about:
   - Local businesses and services
   - Product recommendations from local stores
   - Municipal services and information
   - Community resources and facilities

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
                    with st.spinner("📊 Fetching video stats..."):
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
                    with st.spinner("📝 Fetching video transcript..."):
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
                    with st.spinner("🧠 Analyzing video content..."):
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
                with st.spinner("📝 Fetching video transcript..."):
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
                with st.spinner("📊 Fetching video stats..."):
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
        
        @agent.tool
        async def get_neighborhood_info(ctx: RunContext[None], query: str) -> str:
            """Get information about local businesses, services, and community resources.
            
            Args:
                query: The user's question about the neighborhood
            
            Returns:
                Information about local businesses, services, or community resources
            """
            try:
                # Initialize neighborhood dependencies
                deps = NeighborhoodDeps.create()
                
                # Call neighborhood agent with the query
                with st.spinner("🏘️ Searching neighborhood information..."):
                    result = await neighborhood_agent.run(query, deps=deps)
                
                if isinstance(result.data, NeighborhoodInfo):
                    # Format the response with sources and confidence scores
                    response_parts = []
                    response_parts.append("## Neighborhood Information\n")
                    response_parts.append(result.data.answer)
                    
                    if result.data.sources:
                        response_parts.append("\n\n**Sources:**")
                        for source, score in zip(result.data.sources, result.data.confidence_scores):
                            response_parts.append(f"- {source} (confidence: {score:.2f})")
                    
                    return "\n".join(response_parts)
                else:
                    # Handle NoInformation case
                    return f"## No Information Available\n\n{result.data.message}"
                    
            except Exception as e:
                logfire.error("Failed to get neighborhood information",
                    error=str(e),
                    error_type=type(e).__name__
                )
                return "I apologize, but I encountered an error while retrieving neighborhood information. Please try again or rephrase your question."

        return agent

    agent = get_agent()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a default welcome message
        welcome_msg = loop.run_until_complete(agent.run("Say hi and briefly explain what you can do"))
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
                with st.spinner("🤔 Thinking..."):
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

            # Run the async process using the event loop
            full_response = loop.run_until_complete(process_stream())
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response}) 