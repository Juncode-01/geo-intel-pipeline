"""
Interactive habitat question interface.
Usage: python -m pipeline.ask
"""

from agents.query_agent import QueryAgent

def main():
    agent = QueryAgent()
    print("\nUcluelet Habitat Query System")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        agent.ask_habitat_question(question)
        print()

if __name__ == "__main__":
    main()