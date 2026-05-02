import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import rag_search, web_search

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant for Debales AI.

You have access to two tools:
1. `rag_search` — Use this for any question about Debales AI (products, features,
   pricing, integrations, blog posts, team, company info, etc.)
2. `web_search` — Use this for general questions NOT about Debales AI.

For mixed questions (for example, "How does Debales AI compare to competitor X?"),
use both tools: `rag_search` for the Debales part and `web_search` for the rest.

If you cannot find a confident answer from the available tool results, say so clearly.
Do not invent facts. Prefer citing the retrieved sources when useful.
"""


def _build_llm() -> ChatOllama:
    primary_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "glm-4.6:cloud")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        llm = ChatOllama(model=primary_model, base_url=base_url, temperature=0)
        llm.invoke("ping")
        return llm
    except Exception:
        return ChatOllama(model=fallback_model, base_url=base_url, temperature=0)


def build_agent():
    """Build and compile the LangGraph agent."""
    tools = [rag_search, web_search]
    llm_with_tools = _build_llm().bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
