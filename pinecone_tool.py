import os
from typing import Any, Dict, List, Optional

import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class PineconeTool(BaseModel):
    """A tool for interacting with Pinecone vector database."""
    
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(..., description="Name of the Pinecone index")
    dimension: int = Field(1536, description="Dimension of vectors to store")
    metric: str = Field("cosine", description="Distance metric to use")
    
    _pc: Optional[Pinecone] = None
    _index: Optional[Any] = None
    
    def initialize(self) -> None:
        """Initialize the Pinecone client and index."""
        if not self._pc:
            self._pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists, if not create it
            existing_indexes = [index.name for index in self._pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self._pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            self._index = self._pc.Index(self.index_name)

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> Dict[str, Any]:
        """Upsert vectors into the index."""
        if not self._index:
            self.initialize()
        return self._index.upsert(vectors=vectors, namespace=namespace)
    
    def query(self, 
             vector: List[float], 
             top_k: int = 5, 
             namespace: str = "",
             include_metadata: bool = True,
             include_values: bool = True,
             filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the index for similar vectors.
        
        Args:
            vector: Query vector to find similar vectors for
            top_k: Number of results to return (max 10,000)
            namespace: Namespace to query in (default: "")
            include_metadata: Whether to include metadata in response
            include_values: Whether to include vector values in response  
            filter: Metadata filter expression
            
        Returns:
            Dict containing matches and other query metadata. If no matches found,
            returns {"matches": [], "namespace": namespace}
        """
        if not self._index:
            self.initialize()
            
        try:
            response = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                include_values=include_values,
                filter=filter
            )
            
            # Ensure we always return a valid response structure
            if not response:
                return {"matches": [], "namespace": namespace}
                
            # Ensure matches is always a list
            if "matches" not in response:
                response["matches"] = []
                
            return response
            
        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            # Return a valid empty response structure
            return {"matches": [], "namespace": namespace}
    
    def delete(self, 
               ids: Optional[List[str]] = None, 
               namespace: str = "",
               delete_all: bool = False,
               filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Delete vectors from the index."""
        if not self._index:
            self.initialize()
        return self._index.delete(
            ids=ids,
            namespace=namespace,
            delete_all=delete_all,
            filter=filter
        )
    
    def describe_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if not self._index:
            self.initialize()
        return self._index.describe_index_stats()

def create_pinecone_tool() -> PineconeTool:
    """Create and return a configured PineconeTool instance."""
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")
    
    if not api_key or not environment:
        raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in environment variables")
    
    return PineconeTool(
        api_key=api_key,
        environment=environment,
        index_name=index_name
    ) 