from langchain_core.messages import HumanMessage

from agent import build_agent


def run_cli() -> None:
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
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": conversation_history})

        final_message = result["messages"][-1]
        print(f"\nAssistant: {final_message.content}")

        conversation_history = result["messages"]


if __name__ == "__main__":
    run_cli()

