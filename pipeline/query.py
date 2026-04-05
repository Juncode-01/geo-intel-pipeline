"""
Query Pipeline
--------------
Runs habitat analysis questions against the ChromaDB vector store
using the Gemini LLM. Run this after pipeline/ingest.py has
populated the vector store.
"""

from agents.query_agent import QueryAgent


def run_chanterelle_analysis():
    """Full chanterelle habitat analysis."""
    agent = QueryAgent()
    agent.analyse_chanterelle_habitat()


def ask(question, custom_queries=None):
    """Ask any habitat question."""
    agent = QueryAgent()
    if custom_queries:
        agent.custom_query(question, custom_queries)
    else:
        agent.ask_habitat_question(question)


if __name__ == "__main__":
    run_chanterelle_analysis()