import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import logfire
import nest_asyncio
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (ModelRequest, ModelResponse,
                                  SystemPromptPart, TextPart, UserPromptPart)
from pydantic_ai.usage import Usage, UsageLimits
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

from api_key_manager import APIKey, InMemoryAPIKeyManager
from neighborhood_agent import (NeighborhoodDeps, NeighborhoodInfo,
                                NoInformation, neighborhood_agent)
from youtube_researcher import (VideoInfo, VideoStats, format_video_info,
                                get_video_info, save_transcript_to_temp)

# Load environment variables
load_dotenv()

# Configure logfire for debugging
logfire.configure()

# Create FastAPI app
app = FastAPI(
    title="AI Chat Assistant API",
    description="API for interacting with the AI Chat Assistant",
    version="1.0.0"
)

# Constants for rate limiting and conversation expiry
RATE_LIMIT_WINDOW = 60  # 1 minute window
MAX_REQUESTS_PER_WINDOW = 10  # 10 requests per minute
CONVERSATION_EXPIRY_HOURS = 24  # Conversations expire after 24 hours
MAX_TOKENS = 4000  # Maximum tokens per conversation

# Initialize conversation memory and metadata
conversation_history: Dict[str, List[dict]] = {}
conversation_timestamps: Dict[str, datetime] = {}
conversation_rate_limits: DefaultDict[str, List[float]] = defaultdict(list)
conversation_usage: Dict[str, Usage] = {}

# Initialize API key manager
api_key_manager = InMemoryAPIKeyManager()

# Create initial admin API key
async def create_initial_admin_key() -> str:
    """Create the initial admin API key if none exists."""
    keys = await api_key_manager.get_user_keys("admin")
    if not keys:
        api_key = await api_key_manager.create_key("admin", is_admin=True)
        print("\n=== Initial Admin API Key ===")
        print(f"Key: {api_key.key}")
        print("Please save this key securely - it won't be shown again")
        print("===============================\n")
        return api_key.key
    return None

# Initialize specialized agents
youtube_analysis_agent = Agent(
    'claude-3-5-sonnet-latest',
    system_prompt="""You are a YouTube video research assistant. Analyze the video content and provide:
1. A concise summary
2. Key points and insights
3. Main themes and topics
Format your response in markdown."""
)

# Initialize the main agent with delegation capabilities
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

# Tool for YouTube video analysis with agent delegation
@agent.tool
async def analyze_youtube_video(ctx: RunContext[None], url: str) -> str:
    """Analyze a YouTube video and provide insights using a specialized agent.
    
    Args:
        url: The YouTube video URL to analyze
    
    Returns:
        A detailed analysis of the video content
    """
    try:
        # First try to get video stats
        stats = None
        try:
            logfire.info("Fetching video stats", url=url)
            video_info = await get_video_info(url, fetch_stats=True, should_fetch_transcript=False)
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
            video_info = await get_video_info(url, should_fetch_transcript=True)
            
            # Format video information for analysis
            formatted_info = format_video_info(video_info, include_full_transcript=True)
            
            # Delegate analysis to specialized agent
            result = await youtube_analysis_agent.run(
                f"Analyze this video and provide a comprehensive but concise summary:\n{formatted_info}",
                usage=ctx.usage  # Pass usage context to track tokens
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

# Tool for neighborhood information with agent delegation
@agent.tool
async def get_neighborhood_info(ctx: RunContext[None], query: str) -> str:
    """Get information about local businesses, services, and community resources using a specialized agent.
    
    Args:
        query: The user's question about the neighborhood
    
    Returns:
        Information about local businesses, services, or community resources
    """
    try:
        # Initialize neighborhood dependencies
        deps = NeighborhoodDeps.create()
        
        # Delegate to neighborhood agent
        result = await neighborhood_agent.run(
            query, 
            deps=deps,
            usage=ctx.usage  # Pass usage context to track tokens
        )
        
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

# API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key")

# Request and response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime

class CreateKeyRequest(BaseModel):
    username: str
    is_admin: bool = False

class Message(BaseModel):
    role: str
    content: str

async def get_api_key(api_key: str = Security(api_key_header)) -> Tuple[str, bool]:
    """Validate API key and return tuple of (username, is_admin)."""
    result = await api_key_manager.validate_key(api_key)
    if not result:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or inactive API key"
        )
    return result

async def admin_only(
    credentials: Tuple[str, bool] = Depends(get_api_key)
) -> str:
    """Dependency to ensure only admin users can access an endpoint."""
    username, is_admin = credentials
    if not is_admin:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return username

def get_conversation_history(conversation_id: str) -> List[ModelRequest | ModelResponse]:
    """Convert conversation history to proper format for the agent."""
    if conversation_id not in conversation_history:
        return []
    
    history = []
    # Add system prompt first
    history.append(ModelRequest(parts=[
        SystemPromptPart(content='Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.')
    ]))
    
    # Add conversation history
    for msg in conversation_history[conversation_id]:
        if msg["role"] == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    
    return history

def cleanup_expired_conversations():
    """Remove conversations that have expired based on CONVERSATION_EXPIRY_HOURS."""
    current_time = datetime.now()
    expired_conversations = [
        conv_id for conv_id, timestamp in conversation_timestamps.items()
        if current_time - timestamp > timedelta(hours=CONVERSATION_EXPIRY_HOURS)
    ]
    
    for conv_id in expired_conversations:
        del conversation_history[conv_id]
        del conversation_timestamps[conv_id]
        if conv_id in conversation_rate_limits:
            del conversation_rate_limits[conv_id]

def check_rate_limit(conversation_id: str) -> None:
    """Check if the conversation has exceeded its rate limit.
    
    Args:
        conversation_id: The ID of the conversation to check
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    current_time = time.time()
    # Remove timestamps outside the current window
    conversation_rate_limits[conversation_id] = [
        ts for ts in conversation_rate_limits[conversation_id]
        if current_time - ts < RATE_LIMIT_WINDOW
    ]
    
    if len(conversation_rate_limits[conversation_id]) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_WINDOW} requests per {RATE_LIMIT_WINDOW} seconds."
        )
    
    # Add current request timestamp
    conversation_rate_limits[conversation_id].append(current_time)

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    credentials: Tuple[str, bool] = Depends(get_api_key)
) -> ChatResponse:
    """Process a chat message and return the agent's response."""
    username, _ = credentials
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or f"conv-{datetime.now().isoformat()}"
    
    try:
        # Clean up expired conversations
        cleanup_expired_conversations()
        
        # Check rate limit
        check_rate_limit(conversation_id)
        
        # Initialize conversation history and usage if needed
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
            conversation_timestamps[conversation_id] = datetime.now()
            conversation_usage[conversation_id] = Usage()
        else:
            # Update last activity timestamp
            conversation_timestamps[conversation_id] = datetime.now()
        
        # Add user message to history
        conversation_history[conversation_id].append({
            "role": "user",
            "content": request.message
        })
        
        # Get formatted conversation history
        history = get_conversation_history(conversation_id)
        
        # Set usage limits
        usage_limits = UsageLimits(
            request_limit=MAX_REQUESTS_PER_WINDOW,
            total_tokens_limit=MAX_TOKENS
        )
        
        # Run the agent with the user's message and history
        result = await agent.run(
            request.message, 
            message_history=history,
            usage=conversation_usage[conversation_id],
            usage_limits=usage_limits
        )
        
        # Add assistant response to history
        conversation_history[conversation_id].append({
            "role": "assistant",
            "content": result.data
        })
        
        # Create response
        response = ChatResponse(
            response=result.data,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/conversations/{conversation_id}/history", response_model=List[Message])
async def get_chat_history(
    conversation_id: str,
    credentials: Tuple[str, bool] = Depends(get_api_key)
) -> List[Message]:
    """Get the chat history for a conversation."""
    if conversation_id not in conversation_history:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return [Message(**msg) for msg in conversation_history[conversation_id]]

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    credentials: Tuple[str, bool] = Depends(get_api_key)
) -> dict:
    """Delete a conversation history."""
    if conversation_id not in conversation_history:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    del conversation_history[conversation_id]
    return {"message": "Conversation deleted successfully"}

@app.post("/api-keys", response_model=APIKey)
async def create_api_key(
    request: CreateKeyRequest,
    username: str = Depends(admin_only)
) -> APIKey:
    """Create a new API key for a user. Admin only."""
    return await api_key_manager.create_key(request.username, request.is_admin)

@app.get("/api-keys", response_model=list[APIKey])
async def list_api_keys(
    username: str = Depends(admin_only)
) -> list[APIKey]:
    """List all API keys. Admin only."""
    return await api_key_manager.get_user_keys(username)

@app.delete("/api-keys/{key}")
async def revoke_api_key(
    key: str,
    username: str = Depends(admin_only)
) -> dict:
    """Revoke an API key. Admin only."""
    success = await api_key_manager.revoke_key(key)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="API key not found"
        )
    
    return {"message": "API key revoked successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    # Create initial admin key
    import asyncio
    asyncio.run(create_initial_admin_key())
    
    # Configure uvicorn to use standard asyncio loop
    config = uvicorn.Config(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio"  # Use standard asyncio instead of uvloop
    )
    server = uvicorn.Server(config)
    server.run() 