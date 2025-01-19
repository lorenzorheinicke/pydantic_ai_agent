from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Tuple

from pydantic import BaseModel


class APIKey(BaseModel):
    """Model representing an API key and its metadata."""
    key: str
    username: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    is_admin: bool = False


class APIKeyManager(ABC):
    """Abstract base class for API key management."""
    
    @abstractmethod
    async def validate_key(self, api_key: str) -> Optional[Tuple[str, bool]]:
        """Validate an API key and return tuple of (username, is_admin) if valid."""
        pass
    
    @abstractmethod
    async def create_key(self, username: str, is_admin: bool = False) -> APIKey:
        """Create a new API key for a user."""
        pass
    
    @abstractmethod
    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        pass
    
    @abstractmethod
    async def get_user_keys(self, username: str) -> list[APIKey]:
        """Get all API keys for a user."""
        pass


class InMemoryAPIKeyManager(APIKeyManager):
    """In-memory implementation of API key management."""
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        
        # Initialize with a default admin key (for development only)
        default_key = APIKey(
            key="your-secret-key",
            username="admin",
            created_at=datetime.now(),
            is_active=True,
            is_admin=True  # Mark as admin key
        )
        self._keys[default_key.key] = default_key
    
    async def validate_key(self, api_key: str) -> Optional[Tuple[str, bool]]:
        """Validate an API key and return tuple of (username, is_admin) if valid."""
        if api_key not in self._keys:
            return None
            
        key_data = self._keys[api_key]
        if not key_data.is_active:
            return None
            
        # Update last used timestamp
        key_data.last_used_at = datetime.now()
        return (key_data.username, key_data.is_admin)
    
    async def create_key(self, username: str, is_admin: bool = False) -> APIKey:
        """Create a new API key for a user."""
        import secrets

        # Generate a secure random key
        new_key = APIKey(
            key=secrets.token_urlsafe(32),
            username=username,
            created_at=datetime.now(),
            is_active=True,
            is_admin=is_admin
        )
        
        self._keys[new_key.key] = new_key
        return new_key
    
    async def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key not in self._keys:
            return False
            
        self._keys[api_key].is_active = False
        return True
    
    async def get_user_keys(self, username: str) -> list[APIKey]:
        """Get all API keys for a user."""
        return [
            key for key in self._keys.values()
            if key.username == username
        ] 