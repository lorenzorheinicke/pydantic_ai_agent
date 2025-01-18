import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_google_auth import Authenticate

# Load environment variables
load_dotenv()

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
ALLOWED_DOMAINS = os.getenv('ALLOWED_DOMAINS', '').split(',')
ALLOWED_EMAILS = os.getenv('ALLOWED_EMAILS', '').split(',')
COOKIE_KEY = os.getenv('COOKIE_KEY')

# Create credentials file if it doesn't exist
def ensure_credentials_file():
    """Create the Google credentials file if it doesn't exist"""
    creds_path = Path("google_credentials.json")
    if not creds_path.exists():
        credentials = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": ["http://localhost:8501"],
                "javascript_origins": ["http://localhost:8501"]
            }
        }
        with open(creds_path, "w") as f:
            json.dump(credentials, f)
    return str(creds_path)

def is_email_allowed(email: str) -> bool:
    """Check if the email is allowed based on domain or explicit email address"""
    if not (ALLOWED_DOMAINS or ALLOWED_EMAILS):
        return True  # If no restrictions set, allow all
    
    # Check if email is explicitly allowed
    if email in ALLOWED_EMAILS:
        return True
    
    # Check if email domain is allowed
    domain = email.split('@')[-1]
    return domain in ALLOWED_DOMAINS

def logout():
    """Clear the session state and log out the user"""
    if 'connected' in st.session_state:
        del st.session_state['connected']
    if 'user_info' in st.session_state:
        del st.session_state['user_info']
    if 'auth_error' in st.session_state:
        del st.session_state['auth_error']

def check_auth():
    """Main authentication function that handles the OAuth flow"""
    # Ensure credentials file exists
    creds_path = ensure_credentials_file()
        
    # Initialize authenticator with credentials file
    authenticator = Authenticate(
        secret_credentials_path=creds_path,
        cookie_name='pydantic_ai_auth',
        cookie_key=COOKIE_KEY,
        redirect_uri='http://localhost:8501'
    )

    # Check authentication status
    authenticator.check_authentification()
    
    # Create login button if not authenticated
    if not st.session_state.get('connected'):
        authenticator.login()
        return None
    
    # Check email restrictions
    user_email = st.session_state['user_info'].get('email')
    if not is_email_allowed(user_email):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.error(f"Access denied. Your email ({user_email}) is not authorized to use this application.")
        with col2:
            if st.button("Logout"):
                authenticator.logout()
                st.rerun()
        return None
    
    return st.session_state.get('user_info') 