# Rant to Rock – Obsidian Companion Webapp

A web application that transforms unstructured audio recordings or text transcripts into semantically organized Markdown files for Obsidian.

## Project Overview

Rant to Rock takes unstructured content (audio recordings or existing transcripts) and processes them into well-structured, interconnected Markdown files that follow Obsidian syntax. The application facilitates the creation of visually appealing, interconnected mind maps within an Obsidian vault.

## Modules

### Transcription & Audio Handling (Completed)

The `transcription.py` module handles:
- Processing audio files with preprocessing (noise reduction, normalization)
- Integrating with transcription engines (e.g., Whisper)
- Validating and processing text files
- Extracting and storing metadata (timestamps, confidence scores)

### Context-Aware Chunking & Embedding Generation (Completed)

The `chunking.py` module handles:
- Segmentation of transcripts using a sliding window algorithm
- Preserving context with overlapping segments
- Respecting natural language boundaries when creating chunks
- Generating embeddings for each chunk using NLP models (sentence-transformers)
- Storing chunks and their metadata for further processing

### Semantic Clustering & Topic Grouping (Completed)

The `clustering.py` module handles:
- Grouping related chunks based on semantic similarity
- Implementing hierarchical clustering algorithms
- Considering temporal proximity in clustering decisions
- Optimizing cluster boundaries for coherent topics
- Storing cluster data for summarization

### Summarization & Markdown Generation (Completed)

The `summarization.py` module handles:
- Generating concise summaries for each cluster
- Extracting key entities and topics from content
- Creating well-formatted Markdown files with YAML frontmatter
- Implementing proper Obsidian syntax for links and references
- Organizing content with appropriate headings and sections

### Folder Structure & Export Packaging (Completed)

The `export_packaging.py` module handles:
- Organizing Markdown files into a logical folder structure
- Creating folder hierarchies based on topics, entities, and dates
- Generating index files for easy navigation
- Handling long filenames and special characters
- Bundling files into a ZIP archive for easy import into Obsidian

### Web Interface & API (Completed)

The `main.py` file includes a FastAPI server that provides:
- API endpoints for file upload, status checking, and result retrieval
- CORS middleware for cross-origin requests from the UI
- Background task processing for long-running operations

The `ui/` directory contains a React-based user interface that:
- Provides a drag-and-drop interface for file uploads
- Displays real-time processing status with progress indicators
- Shows previews of generated content (clusters, entities, folder structure)
- Enables downloading the final ZIP archive

## Setup and Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd rant-to-rock
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   # Copy .env.example to .env and edit with your API key
   cp .env.example .env
   # Edit the .env file with your actual OpenAI API key
   ```

   Alternatively, set environment variables directly:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## API Key and Cost Monitoring

The application uses the OpenAI API for generating embeddings with the text-embedding-3-large model. To protect against unexpected costs:

1. **API Key Configuration**: 
   - Store your API key in a `.env` file (never commit this file to version control)
   - See `.env.example` for the required format

2. **Cost Monitoring**:
   - The system estimates token usage before making API calls
   - Warnings are logged when token usage exceeds the configured threshold
   - Default threshold is 10,000 tokens, adjust in your `.env` file

3. **Token Usage Estimation**:
   - The system estimates approximately 0.25 tokens per character in your text
   - Actual usage may vary based on the text content and languages used

**Important**: Processing very large volumes of text may incur significant API costs. Always monitor the logs for warnings about high token usage.

## Running Tests

Execute tests using pytest:
```
pytest
```

## Running the Application

### Command Line Mode

You can run the full pipeline using the main.py script:

```bash
# Process an audio file
python main.py --audio path/to/audio.mp3 --output-dir output --organize-by-topic --organize-by-entity

# Process a text file
python main.py --text path/to/text.txt --output-dir output --organize-by-date

# Process an existing transcript
python main.py --transcript path/to/transcript.json --output-dir output
```

### Web Application Mode

To run the application with the web interface:

1. **Start the development environment**:
   ```bash
   # This will start both the backend API server and the frontend UI
   ./start_dev.sh
   ```

   This script starts:
   - Backend API server on http://localhost:8000
   - Frontend UI on http://localhost:3000

2. **Start the backend and frontend separately**:
   
   Backend:
   ```bash
   # Start the backend API server
   export API_MODE=true
   python main.py
   ```

   Frontend:
   ```bash
   # Navigate to the UI directory
   cd ui
   
   # Install dependencies (first time only)
   npm install
   
   # Start the development server
   npm run dev
   ```

3. **Access the web application**:
   - Open your browser and navigate to http://localhost:3000
   - Upload an audio file or text transcript
   - Monitor the processing status
   - Preview and download the generated Obsidian bundle

### Command Line Options

- Input options (one required):
  - `--audio`: Path to audio file for processing
  - `--text`: Path to text file for processing
  - `--transcript`: Path to existing transcript JSON file

- Output options:
  - `--output-dir`: Directory for output files (default: "output")
  - `--organized-dir`: Directory for organized output files
  - `--archive-path`: Path for ZIP archive

- Configuration options:
  - `--max-chunk-size`: Maximum chunk size in characters (default: 1000)
  - `--min-chunk-size`: Minimum chunk size in characters (default: 100)
  - `--overlap-size`: Overlap size between chunks (default: 200)
  - `--embedding-model`: Embedding model to use (default: "text-embedding-3-large")
  - `--clustering-algorithm`: Clustering algorithm to use (default: "hierarchical")
  - `--summarization-model`: Summarization model to use (default: "gpt-4o")
  - `--max-filename-length`: Maximum filename length (default: 100)

- Organization options:
  - `--organize-by-topic`: Organize files by topic
  - `--organize-by-entity`: Organize files by entity
  - `--organize-by-date`: Organize files by date

## Project Structure

```
.
├── design_document.md       # Detailed design documentation
├── main.py                  # Main script to run the full pipeline
├── start_dev.sh             # Script to start both backend and frontend
├── src/                     # Source code directory
│   ├── __init__.py          # Package initialization
│   ├── transcription.py     # Audio/text processing and transcription module
│   ├── chunking.py          # Context-aware chunking and embedding generation
│   ├── clustering.py        # Semantic clustering and topic grouping
│   ├── summarization.py     # Summarization and Markdown generation
│   └── export_packaging.py  # Folder structure and export packaging
├── ui/                      # Frontend UI code
│   ├── public/              # Static assets
│   ├── src/                 # UI source code
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── lib/             # Utility functions
│   │   ├── pages/           # Page components
│   │   └── types/           # TypeScript type definitions
│   ├── package.json         # UI dependencies
│   └── .env.local           # Environment variables for UI
├── tests/                   # Test directory
│   ├── __init__.py          # Test package initialization
│   ├── test_transcription.py # Unit tests for the transcription module
│   ├── test_chunking.py     # Unit tests for the chunking module
│   ├── test_clustering.py   # Unit tests for the clustering module
│   ├── test_summarization.py # Unit tests for the summarization module
│   └── test_export_packaging.py # Unit tests for the export packaging module
├── integration_test.py      # Integration tests for the full pipeline
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation setup
├── .env.example             # Example environment variables file
└── README.md                # This file
```

## License

[License Information]

## Contact

[Contact Information]