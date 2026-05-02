import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from serpapi import GoogleSearch

load_dotenv()

COLLECTION_NAME = "debales_ai"


def get_retriever():
    """Load the persisted Chroma store and return a retriever."""
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _format_rag_docs(docs: list[Document]) -> str:
    formatted = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        title = doc.metadata.get("title") or source
        formatted.append(
            f"[Result {index}] {title}\nSource: {source}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def rag_search(query: str) -> str:
    """
    Search the Debales AI knowledge base for information about Debales AI,
    its products, features, integrations, pricing, team, or blog content.
    Use this for any question specifically about Debales AI.
    """
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
    except Exception as exc:
        return (
            "RAG search failed. Make sure the vector store exists and Ollama embeddings "
            f"are available. Error: {exc}"
        )

    if not docs:
        return "No relevant information found in the Debales AI knowledge base."

    return _format_rag_docs(docs)


@tool
def web_search(query: str) -> str:
    """
    Search the web using Google for general questions not related to Debales AI.
    Use this for current events, general knowledge, or any non-Debales topic.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "SERPAPI_API_KEY not set. Cannot perform web search."

    try:
        search = GoogleSearch(
            {
                "q": query,
                "api_key": api_key,
                "num": 5,
            }
        )
        results = search.get_dict()
    except Exception as exc:
        return f"Web search failed: {exc}"

    organic = results.get("organic_results", [])
    if not organic:
        return "No web search results found."

    output = []
    for result in organic[:4]:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")
        output.append(f"**{title}**\n{snippet}\nSource: {link}")

    return "\n\n".join(output)

