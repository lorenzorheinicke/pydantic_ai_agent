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

## Prerequisites

- Python 3.6+
- Supadata.ai API key
- OpenAI API key (for Claude integration)
- Streamlit (for web interface)

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
pip install pydantic pydantic-ai python-dotenv requests rich streamlit
```

4. Create a `.env` file in the project root and add your API keys:

```
SUPADATA_API_KEY=your_supadata_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

You can use the tool in two ways:

### 1. Web Interface (Recommended)

Run the Streamlit web interface:

```bash
streamlit run streamlit_chat.py
```

Then open your browser and navigate to the displayed URL (typically http://localhost:8501)

### 2. Command Line Interface

Run the YouTube researcher in terminal mode:

```bash
python youtube_researcher.py
```

## Project Structure

- `youtube_researcher.py`: Main application file
- `streamlit_chat.py`: Streamlit web interface
- `docs/`: API documentation
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

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
