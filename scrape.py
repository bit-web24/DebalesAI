import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

URLS_FILE = "urls.txt"
COLLECTION_NAME = "debales_ai"


def load_urls(filepath: str) -> list[str]:
    """Read URLs from a text file, skipping blanks and comments."""
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    urls = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        raise ValueError(f"No URLs found in {filepath}. Add at least one URL.")
    return urls


def build_vectorstore() -> None:
    """Load pages, chunk them, embed them, and persist them in ChromaDB."""
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
        raise RuntimeError("No pages loaded. Check URLs or internet connectivity.")

    print(f"Loaded {len(docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("No chunks were created from the scraped documents.")

    print(f"Created {len(chunks)} chunks.")
    print("Embedding and saving to ChromaDB...")

    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=COLLECTION_NAME,
    )
    print("Vector store built successfully.")


if __name__ == "__main__":
    build_vectorstore()
