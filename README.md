# YouTube Research Assistant

A Python-based tool that leverages AI to analyze YouTube video content through transcripts. This tool allows users to extract insights, summaries, and answer questions about YouTube videos using the power of Claude AI.

## Features

- Extract and process YouTube video transcripts
- Generate comprehensive video summaries
- Interactive Q&A about video content
- Support for multiple languages
- AI-powered analysis using Claude 3.5 Sonnet
- Rich text output formatting

## Prerequisites

- Python 3.6+
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
pip install pydantic pydantic-ai python-dotenv requests rich
```

4. Create a `.env` file in the project root and add your API keys:

```
SUPADATA_API_KEY=your_supadata_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run the YouTube researcher:

```bash
python youtube_researcher.py
```

2. Enter a YouTube URL when prompted

3. Ask questions about the video content

## Project Structure

- `youtube_researcher.py`: Main application file
- `docs/`: API documentation
- `.env`: Environment variables configuration
- `requirements.txt`: Project dependencies

## Environment Variables

- `SUPADATA_API_KEY`: API key for Supadata.ai services
- `OPENAI_API_KEY`: API key for OpenAI/Claude services

## Features in Detail

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
