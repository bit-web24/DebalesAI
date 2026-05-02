# Debales AI Assistant

A CLI chatbot that answers questions about Debales AI using RAG and handles general questions using Google Search via SerpApi.

Built with LangGraph, LangChain, Ollama, and ChromaDB.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- [Ollama](https://ollama.com) installed and running locally
- A [SerpApi](https://serpapi.com) API key

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Pull Ollama models

```bash
ollama pull gpt-oss:120b-cloud
# or use the fallback model
ollama pull glm-4.6:cloud

ollama pull nomic-embed-text
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your `SERPAPI_API_KEY`.

### 4. Build the Debales knowledge base

```bash
uv run python scrape.py
```

This reads `urls.txt`, scrapes the configured Debales pages, and creates a local `chroma_db/` directory.

### 5. Start the chatbot

```bash
uv run python main.py
```

## Example Prompts

| Question | Expected Behavior |
|----------|-------------------|
| What does Debales AI do? | Uses RAG |
| What are Debales AI's integrations? | Uses RAG |
| What is the capital of France? | Uses web search |
| Who is the CEO of OpenAI? | Uses web search |
| How does Debales AI compare to Drift? | Uses both RAG and web search |

## Architecture

User Input  
    |  
    v  
LangGraph Agent (Ollama LLM)  
    |  
    +--> `rag_search` -> ChromaDB (Debales AI scraped content)  
    |  
    +--> `web_search` -> SerpApi (Google Search)  
    |  
    v  
Final Answer (CLI)

## Notes

- The LLM decides which tool or tools to call. There is no hardcoded keyword routing.
- Mixed queries can use both tools in one turn.
- Conversation history is preserved in memory for multi-turn chat.
- If the vector store is missing or a tool cannot answer confidently, the assistant should say so instead of inventing details.

