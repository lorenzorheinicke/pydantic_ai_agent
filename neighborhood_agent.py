import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from pinecone_tool import create_pinecone_tool

# Load environment variables
load_dotenv()

class NeighborhoodDeps(BaseModel):
    """Dependencies for the neighborhood agent."""
    openai_client: OpenAI
    pinecone_tool: Any
    
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod
    def create(cls) -> "NeighborhoodDeps":
        """Create and initialize dependencies."""
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pinecone_tool = create_pinecone_tool()
        pinecone_tool.initialize()
        return cls(openai_client=openai_client, pinecone_tool=pinecone_tool)

class NeighborhoodInfo(BaseModel):
    """Structured response for neighborhood information."""
    answer: str
    sources: List[str]
    confidence_scores: List[float]

class NoInformation(BaseModel):
    """Response when no relevant information is found."""
    message: str = Field(..., description="Message explaining why no information could be provided")

# Create the agent with proper dependencies and result types
neighborhood_agent = Agent[NeighborhoodDeps, Union[NeighborhoodInfo, NoInformation]](
    "openai:gpt-4o-mini",
    deps_type=NeighborhoodDeps,
    result_type=Union[NeighborhoodInfo, NoInformation],  # type: ignore
    system_prompt="""You are a knowledgeable local assistant that helps people find and understand information about:
    1. Local businesses and service providers (restaurants, shops, contractors, professionals, etc.)
    2. Product recommendations from local stores and vendors
    3. Municipal services and information (permits, recycling schedules, city services, etc.)
    4. Community resources and public facilities
    
    Use the provided data from the vector database to give accurate and helpful recommendations and answers. 
    When making recommendations, prioritize local businesses and services that are well-reviewed and established 
    in the community. For municipal services, provide clear, actionable information about processes, requirements, 
    and contact details when available.
    
    When you don't have specific information about something, be honest about it and return a NoInformation response. 
    Always prioritize factual information from the database over general knowledge. If providing business recommendations, 
    try to include relevant details such as location, hours, contact information, and any notable specialties or services."""
)

@neighborhood_agent.tool
async def embed_text(ctx: RunContext[NeighborhoodDeps], text: str) -> List[float]:
    """Generate embeddings for the given text using OpenAI."""
    response = ctx.deps.openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

@neighborhood_agent.tool
async def search_neighborhood_data(
    ctx: RunContext[NeighborhoodDeps], 
    query: str, 
    top_k: int = 5
) -> List[Dict]:
    """Search the Pinecone database for relevant neighborhood information."""
    try:
        query_embedding = await embed_text(ctx, query)
        
        results = ctx.deps.pinecone_tool.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,  # We don't need vector values in the response
            namespace="whatsapp-group"
        )
        
        matches = results.get("matches", [])
        
        # Ensure each match has required fields
        processed_matches = []
        for match in matches:
            processed_match = {
                "id": match.get("id", ""),
                "score": match.get("score", 0.0),
                "metadata": match.get("metadata", {})
            }
            processed_matches.append(processed_match)
            
        return processed_matches
        
    except Exception as e:
        print(f"Error searching neighborhood data: {str(e)}")
        return []  # Return empty list on error

@neighborhood_agent.tool
async def add_neighborhood_data(
    ctx: RunContext[NeighborhoodDeps],
    content: str,
    metadata: Dict
) -> Dict:
    """Add new neighborhood information to the vector database."""
    embedding = await embed_text(ctx, content)
    
    vector = {
        "id": metadata.get("id", os.urandom(12).hex()),
        "values": embedding,
        "metadata": {
            "content": content,
            **metadata
        }
    }
    
    return ctx.deps.pinecone_tool.upsert(vectors=[vector], namespace="whatsapp-group")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize dependencies
        deps = NeighborhoodDeps.create()
        
        # Add some example data
        await neighborhood_agent.run(
            "Add information about Downtown.",
            deps=deps,
            message_history=[{
                "role": "user",
                "content": """Add this neighborhood data:
                The Downtown district is known for its vibrant nightlife, diverse dining options, 
                and easy access to public transportation. The area has seen significant development 
                in recent years, with new luxury apartments and boutique shops opening regularly."""
            }]
        )
        
        # Ask a question
        result = await neighborhood_agent.run(
            "What can you tell me about the Downtown area's nightlife and dining options?",
            deps=deps
        )
        
        if isinstance(result.data, NeighborhoodInfo):
            print("Answer:", result.data.answer)
            print("\nSources:", result.data.sources)
            print("Confidence Scores:", result.data.confidence_scores)
        else:
            print("No information found:", result.data.message)
    
    asyncio.run(main()) 