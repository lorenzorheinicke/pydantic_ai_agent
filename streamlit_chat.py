import asyncio
import os
from datetime import datetime

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import (ModelRequest, ModelResponse,
                                  SystemPromptPart, TextPart, UserPromptPart)

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
        'claude-3-5-sonnet-latest',
        system_prompt='Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.',
    )
    
    @agent.tool_plain
    def get_datetime() -> str:
        """Get the current date and time."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
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
        with st.spinner("Thinking..."):
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
            
            # Get response using message history
            result = asyncio.run(agent.run(prompt, message_history=history))
            st.markdown(result.data)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result.data}) 