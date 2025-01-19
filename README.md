# YouTube Research Assistant

A Python-based tool that leverages AI to analyze YouTube video content through transcripts. This tool allows users to extract insights, summaries, and answer questions about YouTube videos using the power of Claude AI.

## Features

- Extract and process YouTube video transcripts
- Generate comprehensive video summaries
- Interactive Q&A about video content
- Support for multiple languages
- AI-powered analysis using Claude 3.5 Sonnet
- Rich text output formatting
- Web-based interface using Streamlit
- Real-time chat interface for video analysis
- Secure authentication system
- Integration with Pinecone for vector storage
- Neighborhood-based content analysis

## Prerequisites

- Python 3.9+
- Claude API access (via Anthropic)
- YouTube Data API credentials
- Pinecone API key
- Streamlit (for web interface)
- Supadata.ai API key
- OpenAI API key (for Claude integration)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pydantic_ai_agent
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables by copying `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then edit `.env` with your API keys and credentials:

```
SUPADATA_API_KEY=your_supadata_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
YOUTUBE_API_KEY=your_youtube_api_key
# Add other required credentials
```

## Usage

### Web Interface (Recommended)

Run the Streamlit web interface:

```bash
streamlit run streamlit_chat.py
```

Then open your browser and navigate to the displayed URL (typically http://localhost:8501)

## Project Structure

- `youtube_researcher.py`: Main YouTube analysis logic
- `streamlit_chat.py`: Streamlit web interface
- `pinecone_tool.py`: Pinecone vector database integration
- `neighborhood_agent.py`: Neighborhood-based content analysis
- `auth.py`: Authentication system
- `config.py`: Configuration management
- `docs/`: API and component documentation
- `.env`: Environment variables configuration
- `requirements.txt`: Project dependencies

## Features in Detail

### Web Interface

- User-friendly Streamlit-based interface
- Real-time chat interactions
- Easy video URL input
- Persistent chat history
- Markdown-formatted responses

### Video Analysis

- Transcript extraction and processing
- Language detection and support
- Automatic content summarization

### Interactive Q&A

- Context-aware responses
- Access to full transcript when needed
- Rich markdown formatting for responses

## Documentation

Detailed documentation for various components can be found in the `docs/` directory:

- `docs/pinecone.md`: Pinecone integration guide
- `docs/streamlit_authentication.md`: Authentication system documentation
- `docs/youtube_data_api.md`: YouTube API setup guide
- `docs/pydanticai_api.md`: PydanticAI integration
- `docs/supadata_api.md`: Supadata API documentation

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
