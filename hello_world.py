import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables from .env file
load_dotenv()

console = Console()
agent = Agent(  
    'claude-3-5-sonnet-latest',
    system_prompt='Be concise and format your response in markdown. Use markdown features like **bold**, *italics*, `code`, lists, and other formatting where appropriate.',
)

user_input = "Hello!"
result = agent.run_sync(user_input)  
console.print(Markdown(result.data))
while True:
    user_input = input("> ")
    if user_input.lower() in ['quit', 'exit']:
        console.print(Markdown("**Goodbye!**"))
        break
    result = agent.run_sync(user_input, message_history=result.all_messages())
    console.print(Markdown(result.data))
