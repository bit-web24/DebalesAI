# Debales AI Assistant — Implementation Spec

## Overview

Build a CLI chatbot that acts as a Debales AI assistant using LangGraph. The agent uses:
- **RAG** for Debales-related questions (scraped from their website)
- **SERP API** for general web questions
- **Ollama** as the local LLM — `gpt-oss:120b-cloud` (primary) or `glm-4.6:cloud` (fallback)
- **LangGraph** to wire the workflow together

The LLM decides which tool to call (or both) based on the question — no hardcoded keyword routing.

---

## Project Structure

```
debales-agent/
├── main.py              # CLI entry point
├── scrape.py            # Scrape Debales AI website and build vector store
├── agent.py             # LangGraph agent definition
├── tools.py             # RAG tool + SERP tool definitions
├── urls.txt             # One URL per line — edit this to control what gets scraped
├── pyproject.toml       # Project metadata and dependencies (managed by uv)
├── .env.example
└── README.md
```

---

## Tech Stack

| Component       | Library / Service                          |
|-----------------|--------------------------------------------|
| LLM             | Ollama (`gpt-oss:120b-cloud` / `glm-4.6:cloud`) |
| Embeddings      | Ollama (`nomic-embed-text`)                     |
| Vector Store    | ChromaDB (local, persistent)                    |
| Web Scraping    | LangChain `WebBaseLoader` (uses bs4 internally) |
| SERP API        | SerpApi (`google-search-results` package)  |
| Agent Framework | LangGraph + LangChain                      |
| Env vars        | `python-dotenv`                            |

---

## Step 1 — `pyproject.toml`

Created by running `uv init debales-agent` then `uv add <packages>`. The final file should look like this:

```toml
[project]
name = "debales-agent"
version = "0.1.0"
description = "Debales AI assistant using LangGraph, RAG, and SerpApi"
requires-python = ">=3.10"
dependencies = [
    "langgraph",
    "langchain",
    "langchain-community",
    "langchain-ollama",
    "langchain-chroma",
    "chromadb",
    "beautifulsoup4",
    "lxml",
    "google-search-results",
    "python-dotenv",
]
```

---

## Step 2 — `.env.example`

```
SERPAPI_API_KEY=your_serpapi_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:120b-cloud
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./chroma_db
```

---

## Step 3 — `urls.txt`

One URL per line. Blank lines and lines starting with `#` are ignored.
Edit this file freely — no code changes needed to add or remove pages.

```
# Debales AI — main pages
https://debales.ai
https://debales.ai/about
https://debales.ai/product
https://debales.ai/integrations
https://debales.ai/pricing
https://debales.ai/contact

# Blog posts — add individual post URLs here
https://debales.ai/blog
```

---

## Step 4 — `scrape.py`

This script reads `urls.txt`, loads each page via WebBaseLoader, and builds the ChromaDB vector store.
Run it once before starting the chatbot: `python scrape.py`

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

URLS_FILE = "urls.txt"


def load_urls(filepath: str) -> list[str]:
    """Read URLs from a text file. Ignores blank lines and # comments."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    urls = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        raise ValueError(f"No URLs found in {filepath}. Add at least one URL.")
    return urls


def build_vectorstore():
    """Load pages via WebBaseLoader, chunk, embed, and save to ChromaDB."""
    urls = load_urls(URLS_FILE)
    print(f"Found {len(urls)} URLs in {URLS_FILE}.")
    print("Loading pages with WebBaseLoader...")

    loader = WebBaseLoader(
        web_paths=urls,
        bs_get_text_kwargs={"separator": "\n", "strip": True},
        requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}, "timeout": 10},
        continue_on_failure=True,
    )

    docs = loader.load()

    if not docs:
        raise RuntimeError("No pages loaded. Check URLs or internet connection.")

    print(f"Loaded {len(docs)} pages. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and saving to ChromaDB (this may take a minute)...")
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name="debales_ai",
    )
    print("Vector store built and saved successfully.")


if __name__ == "__main__":
    build_vectorstore()
```

---

## Step 5 — `tools.py`

Defines both tools the agent can call.

```python
import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()


def get_retriever():
    """Load the ChromaDB vector store and return a retriever."""
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=embeddings,
        collection_name="debales_ai",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@tool
def rag_search(query: str) -> str:
    """
    Search the Debales AI knowledge base for information about Debales AI,
    its products, features, integrations, pricing, team, or blog content.
    Use this for any question specifically about Debales AI.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the Debales AI knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@tool
def web_search(query: str) -> str:
    """
    Search the web using Google for general questions not related to Debales AI.
    Use this for current events, general knowledge, or any non-Debales topic.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "SERPAPI_API_KEY not set. Cannot perform web search."

    search = GoogleSearch({
        "q": query,
        "api_key": api_key,
        "num": 5,
    })
    results = search.get_dict()

    organic = results.get("organic_results", [])
    if not organic:
        return "No web search results found."

    output = []
    for r in organic[:4]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        output.append(f"**{title}**\n{snippet}\nSource: {link}")

    return "\n\n".join(output)
```

---

## Step 6 — `agent.py`

The LangGraph agent. Uses a simple `agent → tools → agent` loop.
The LLM decides which tool to call based on the question.

```python
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import rag_search, web_search

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant for Debales AI.

You have access to two tools:
1. `rag_search` — Use this for any question about Debales AI (products, features, 
   pricing, integrations, blog posts, team, company info, etc.)
2. `web_search` — Use this for general questions NOT about Debales AI.

For mixed questions (e.g. "How does Debales AI compare to competitor X?"), 
use both tools: rag_search for the Debales part, web_search for the rest.

If you cannot find a confident answer from either tool, say so honestly.
Never make up information or hallucinate facts.
"""


def build_agent():
    """Build and return the compiled LangGraph agent."""
    tools = [rag_search, web_search]

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        """The main agent node — calls LLM with tools bound."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # routes to tools or END
    graph.add_edge("tools", "agent")                       # loop back after tool call

    return graph.compile()
```

---

## Step 7 — `main.py`

The CLI chat loop.

```python
from langchain_core.messages import HumanMessage
from agent import build_agent


def run_cli():
    print("=" * 50)
    print("  Debales AI Assistant")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    agent = build_agent()
    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        conversation_history.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": conversation_history})

        # The last message in the result is the final AI response
        final_message = result["messages"][-1]
        print(f"\nAssistant: {final_message.content}")

        # Keep full history for multi-turn context
        conversation_history = result["messages"]


if __name__ == "__main__":
    run_cli()
```

---

## Step 8 — `README.md`

```markdown
# Debales AI Assistant

A CLI chatbot that answers questions about Debales AI using RAG,
and general questions using Google Search (SerpApi).

Built with LangGraph, LangChain, Ollama, and ChromaDB.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- [Ollama](https://ollama.com) installed and running locally
- A [SerpApi](https://serpapi.com) API key (free tier available)

## Setup

### 1. Create and initialise the project

uv init 

### 2. Add dependencies

uv add langgraph langchain langchain-community langchain-ollama langchain-chroma chromadb beautifulsoup4 lxml google-search-results python-dotenv

### 3. Pull Ollama models

ollama pull gpt-oss:120b-cloud
# or if unavailable:
ollama pull glm-4.6:cloud

ollama pull nomic-embed-text

### 4. Configure environment variables

cp .env.example .env
# Edit .env and add your SERPAPI_API_KEY

### 5. Add your URLs to urls.txt, then build the knowledge base

uv run python scrape.py

This only needs to be run once (or re-run when you add new URLs).
It creates a `chroma_db/` folder with the vector store.

### 6. Start the chatbot

uv run python main.py

## Example Prompts

| Question                                 | Expected Behavior         |
|------------------------------------------|---------------------------|
| What does Debales AI do?                 | Uses RAG                  |
| What are Debales AI's integrations?      | Uses RAG                  |
| What is the capital of France?           | Uses Web Search           |
| Who is the CEO of OpenAI?                | Uses Web Search           |
| How does Debales AI compare to Drift?    | Uses both RAG + Web Search|

## Architecture

User Input
    │
    ▼
LangGraph Agent (Ollama LLM)
    │
    ├──► rag_search tool  →  ChromaDB (Debales AI scraped content)
    │
    └──► web_search tool  →  SerpApi (Google Search)
    │
    ▼
Final Answer (CLI)

## Notes

- The LLM decides which tool(s) to call — no hardcoded keyword routing.
- The agent never hallucinates: if neither tool returns useful results, 
  it says so explicitly.
- Multi-turn conversation is supported (context is maintained in memory).
```

---

## Implementation Notes for Claude Code

### Ollama model compatibility
- Primary model: `gpt-oss:120b-cloud`. Fallback: `glm-4.6:cloud`.
- Both models support tool/function calling via Ollama's OpenAI-compatible API.
- Change `OLLAMA_MODEL` in `.env` to switch models without touching code.
- Run `ollama list` to confirm the model is pulled before starting.
- If tool calling fails silently, check that the model is a chat model (not base), and that Ollama is version ≥ 0.3.

### WebBaseLoader notes
- `WebBaseLoader` from `langchain_community.document_loaders` handles HTTP requests and HTML parsing internally using `bs4` and `lxml`.
- Pass `continue_on_failure=True` so one bad URL doesn't crash the whole scrape.
- `split_documents(docs)` is used (not `create_documents()`) because WebBaseLoader already returns `Document` objects — `metadata` (source URL, title) is preserved through chunking automatically.
- The loader fetches all URLs in the list concurrently; each page becomes one `Document` before splitting.
- `Chroma.from_documents(persist_directory=...)` saves automatically.
- On subsequent runs, load with `Chroma(persist_directory=..., embedding_function=...)`.
- Do not call `vectorstore.persist()` — it's deprecated in newer chromadb versions.

### SerpApi import
- The package is `google-search-results` but imported as `from serpapi import GoogleSearch`.

### LangGraph version
- Use `langgraph >= 0.2`. The `tools_condition` import is from `langgraph.prebuilt`.
- `MessagesState` is from `langgraph.graph` and handles message list merging automatically.

### Tool calling with Ollama
- `ChatOllama.bind_tools(tools)` works with function-calling capable models.
- Always set `temperature=0` for deterministic tool routing.
- The `ToolNode` from `langgraph.prebuilt` handles execution and result formatting automatically.

---

## Deliverables Checklist

- [ ] `urls.txt` — list of URLs to scrape (one per line, supports `#` comments)
- [ ] `scrape.py` — reads urls.txt, loads pages, builds ChromaDB
- [ ] `tools.py` — defines `rag_search` and `web_search` tools
- [ ] `agent.py` — LangGraph graph with agent + tools nodes
- [ ] `main.py` — CLI chat loop with conversation history
- [ ] `pyproject.toml` — uv-managed dependencies
- [ ] `.env.example`
- [ ] `README.md` with setup steps
- [ ] Demo video (record a short walkthrough showing RAG, SERP, and mixed queries)
