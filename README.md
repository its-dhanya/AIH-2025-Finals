# WeaveDocs

**A prototype developed for the Adobe India Hackathon Grand Finale by Team Cache Me If You Can**

## Overview

WeaveDocs is a fullstack application that revolutionizes how you interact with document collections. Upload your PDFs and unlock a new way to explore, understand, and consume information through intelligent document analysis, cross-referencing, and audio summaries.

The application leverages advanced AI to extract document structure, identify relationships between content across multiple documents, and provide contextual insights. Whether you're researching, studying, or analyzing large document collections, WeaveDocs transforms static PDFs into an interactive, connected knowledge base.

## Key Features

### 1. **Upload Documents**
Upload a single file or a collection of documents (PDFs).

### 2. **Heading Extraction (Round 1)**
Automatically extract document structure (H1, H2, H3) to build a navigable outline.

### 3. **Jump to Heading**
Click any extracted heading to navigate to that part of the PDF.

### 4. **Cross-Document Relevance (Round 1.B)**
Select any line or paragraph and get the related sections from other PDFs in the same collection using round 1.B retrieval.

### 5. **Navigate to Source**
Click a retrieved section to open the exact page in the original PDF where that section appears.

### 6. **Insights Tab**
Get a concise gist of your highlighted text and the retrieved sections, with examples and contradictions where applicable.

### 7. **Podcast**
Generate an audio overview (podcast-style) of the highlighted section and its relevant contents.

## Prerequisites

- **Docker installed**: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- **API keys** for the external services you use (store them as environment variables when running the container):
  - `ADOBE_EMBED_API_KEY`
  - `GEMINI_API_KEY`
  - Any other provider-specific keys for TTS, etc.

## Quick Start

### Build the Docker Image

```bash
docker build -t my-fullstack-app .
```

### Run the App (Docker)

```bash
docker run -d --platform linux/amd64 \
  --name my-fullstack-app \
  -e ADOBE_EMBED_API_KEY=<your-adobe-key> \
  -e GEMINI_API_KEY=<your-gemini-key> \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_MODEL=gemini-1.5-flash \
  -e TTS_PROVIDER=gtts \
  -p 8080:8080 \
  -p 3000:3000 \
  my-fullstack-app:latest
```

**Access the application:**
- **Backend**: `http://localhost:3000`
- **Frontend**: `http://localhost:8080`

> **Note**: If the frontend page is not loading, refresh it and it will work.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ADOBE_EMBED_API_KEY` | Adobe Embed API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `LLM_PROVIDER` | LLM provider (e.g., `gemini`) |
| `GEMINI_MODEL` | Gemini model |
| `TTS_PROVIDER` | Text-to-Speech provider (e.g., `gtts`) |

## Useful Docker Commands

### View Logs (Combined)
```bash
docker logs -f my-fullstack-app
```

### Enter the Container Shell
```bash
docker exec -it my-fullstack-app sh
```

### View Backend Logs
```bash
docker exec -it my-fullstack-app sh -c "tail -f /var/log/backend.log"
```
*(Adjust path to wherever your backend writes logs)*

### Stop Container
```bash
docker stop my-fullstack-app
```

## Development (Optional)

If you prefer to run locally without Docker, follow these typical steps:

### Backend Setup
1. Create and activate a Python virtual environment for the backend
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend server: `uvicorn backend.api:app --reload --port 8080`

### Frontend Setup
1. Navigate to the `frontend` folder
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

The frontend will typically run on port 8080.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
