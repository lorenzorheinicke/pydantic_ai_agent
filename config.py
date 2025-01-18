import os
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google OAuth settings
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# Load and parse allowed domains from environment variable
_allowed_domains = os.getenv("ALLOWED_DOMAINS", "")
ALLOWED_DOMAINS: List[str] = [domain.strip() for domain in _allowed_domains.split(",") if domain.strip()] if _allowed_domains else []

# Load and parse allowed emails from environment variable
_allowed_emails = os.getenv("ALLOWED_EMAILS", "")
ALLOWED_EMAILS: List[str] = [email.strip() for email in _allowed_emails.split(",") if email.strip()] if _allowed_emails else []

# Cookie settings
COOKIE_NAME = "streamlit_auth"
COOKIE_KEY = os.getenv("COOKIE_KEY", "your-secret-key")  # Change this in production
COOKIE_EXPIRY_DAYS = 30

def is_authorized(email: str) -> bool:
    """Check if the email is authorized to access the app."""
    if not email:
        return False
        
    # Check specific allowed emails
    if email in ALLOWED_EMAILS:
        return True
        
    # Check allowed domains
    domain = email.split("@")[-1]
    if domain in ALLOWED_DOMAINS:
        return True
        
    return False 