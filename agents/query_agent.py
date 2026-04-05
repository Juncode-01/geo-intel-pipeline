"""
Query Agent
-----------
Connects the ChromaDB vector store to a Gemini LLM to answer
geographic and ecological questions grounded in BC government data.

For complex habitat questions like chanterelle distribution,
the agent decomposes the question into multiple focused sub-queries,
retrieves relevant chunks for each, deduplicates and ranks the
combined results, then passes the best context to the LLM.

This is the RAG (Retrieval Augmented Generation) pattern.
"""

import os
from google import genai
from agents.vector_store_agent import VectorStoreAgent
from config import GEMINI_API_KEY

# How many chunks to retrieve per sub-query
CHUNKS_PER_QUERY = 5

# How many total chunks to pass to LLM after dedup and ranking
MAX_CONTEXT_CHUNKS = 15

# Relevance cutoff - ignore chunks below this score
MIN_RELEVANCE = 0.25

# Decomposed queries for chanterelle habitat analysis
CHANTERELLE_HABITAT_QUERIES = [
    "soil moisture drainage surficial material terrain texture",
    "coniferous forest species cedar hemlock douglas-fir spruce",
    "biogeoclimatic zone coastal western hemlock subzone variant",
    "terrain slope aspect elevation topography",
    "old growth ancient forest seral stage stand age",
    "forest inventory species composition canopy cover",
    "aquifer groundwater hydrology watershed drainage basin",
    "bioterrain terrain classification moisture nutrient regime",
    "protected area ecological reserve intact forest",
    "precipitation rainfall moisture regime climate",
    "soil survey organic matter mineral soil type",
    "riparian zone stream buffer forest understorey"
]


class QueryAgent:

    def __init__(self):
        # Configure Gemini
        if not GEMINI_API_KEY:
            raise ValueError(
                "GOOGLE_GEMINI_API_KEY not found in environment. "
                "Check your .env file."
            )

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API configured.\n")

        # Load vector store
        self.vector_store = VectorStoreAgent()

    # ----------------------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------------------

    def retrieve_context(self, queries, n_per_query=CHUNKS_PER_QUERY):
        """
        Run multiple queries against the vector store,
        combine results, deduplicate, and rank by relevance.
        Returns a list of the best unique chunks.
        """
        all_results = {}  # chunk_id -> result dict

        for query in queries:
            results = self.vector_store.search(
                query, n_results=n_per_query
            )

            for r in results:
                chunk_id = r["id"]
                relevance = 1 - r["distance"]

                # Skip low relevance chunks
                if relevance < MIN_RELEVANCE:
                    continue

                # Keep the highest relevance score if seen before
                if chunk_id not in all_results or \
                   relevance > all_results[chunk_id]["relevance"]:
                    all_results[chunk_id] = {
                        "id": chunk_id,
                        "text": r["text"],
                        "metadata": r["metadata"],
                        "relevance": relevance,
                        "matched_query": query
                    }

        # Sort by relevance and take top N
        ranked = sorted(
            all_results.values(),
            key=lambda x: x["relevance"],
            reverse=True
        )[:MAX_CONTEXT_CHUNKS]

        print(f"Retrieved {len(all_results)} unique chunks "
              f"across {len(queries)} queries")
        print(f"Using top {len(ranked)} chunks as context\n")

        return ranked

    def build_context_text(self, chunks):
        """
        Format retrieved chunks into a readable context block
        for the LLM prompt. Include source attribution.
        """
        lines = []
        lines.append(
            "The following information comes from BC government "
            "geospatial datasets covering the Ucluelet area:\n"
        )

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            title = meta.get("title", meta.get("dataset_id", "?"))
            relevance = chunk["relevance"]

            lines.append(f"--- Source {i+1}: {title} "
                        f"(relevance: {relevance:.3f}) ---")
            lines.append(chunk["text"])
            lines.append("")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # LLM QUERY
    # ----------------------------------------------------------

    def ask(self, question, queries=None, verbose=True):
        """
        Answer a question using RAG over the vector store.

        question: the natural language question to answer
        queries: list of sub-queries for retrieval
                 (uses question alone if not provided)
        verbose: print retrieved chunks before LLM response
        """
        # Use provided queries or fall back to the question itself
        retrieval_queries = queries or [question]

        print("=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)
        print()

        # Retrieve relevant context
        print("Retrieving relevant context from vector store...")
        chunks = self.retrieve_context(retrieval_queries)

        if not chunks:
            print("No relevant context found in vector store.")
            return None

        if verbose:
            print("Top chunks retrieved:")
            for i, chunk in enumerate(chunks[:5]):
                meta = chunk["metadata"]
                title = meta.get(
                    "title", meta.get("dataset_id", "?")
                )
                print(f"  [{i+1}] {title[:55]} "
                      f"(relevance: {chunk['relevance']:.3f})")
            print()

        # Build context block
        context = self.build_context_text(chunks)

        # Build the full prompt
        prompt = self._build_prompt(question, context)

        # Send to Gemini
        print("Sending to Gemini...\n")
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
)

        print("=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(response.text)

        return response.text

    def _build_prompt(self, question, context):
        """
        Construct the full prompt for the LLM.
        Instructs it to stay grounded in the provided data
        and be explicit about uncertainty.
        """
        return f"""You are an expert ecological geographer 
specializing in Pacific Northwest mycology and forest ecosystems.

You have been given real BC government geospatial data covering 
the Ucluelet area on Vancouver Island. Your task is to answer 
the question below using ONLY the provided data as your primary 
source. You may draw on general ecological knowledge to interpret 
the data, but clearly distinguish between what the data shows 
and what you are inferring.

Be specific about locations, coordinates, and measurements when 
the data provides them. If the data is insufficient to answer 
part of the question, say so explicitly rather than speculating.

{context}

QUESTION: {question}

Please structure your response as follows:
1. WHAT THE DATA SHOWS — specific findings from the provided sources
2. ECOLOGICAL INTERPRETATION — what this means for the habitat question
3. PREDICTED HABITAT ZONES — your best assessment with confidence level
4. DATA GAPS — what additional data would improve this prediction
5. SUGGESTED SURVEY LOCATIONS — specific coordinates or areas to 
   prioritize for field verification
"""

    # ----------------------------------------------------------
    # CHANTERELLE SPECIFIC ANALYSIS
    # ----------------------------------------------------------

    def analyse_chanterelle_habitat(self):
        """
        Run the full chanterelle habitat analysis using
        decomposed sub-queries for comprehensive retrieval.
        """
        question = (
            "Based on the soil conditions, forest composition, "
            "terrain, hydrology, and biogeoclimatic data available "
            "for the Ucluelet BC area, where are chanterelle "
            "mushrooms (Cantharellus cibarius and related species) "
            "most likely to be found? What specific habitat "
            "conditions in this data support or limit their "
            "distribution? Identify the most promising zones "
            "for field verification surveys."
        )

        return self.ask(
            question=question,
            queries=CHANTERELLE_HABITAT_QUERIES,
            verbose=True
        )

    def ask_habitat_question(self, question):
        """
        Answer any habitat-related question using the
        decomposed chanterelle query set for broad retrieval.
        """
        return self.ask(
            question=question,
            queries=CHANTERELLE_HABITAT_QUERIES,
            verbose=True
        )

    def custom_query(self, question, sub_queries):
        """
        Answer a question with custom sub-queries.
        Use this when the chanterelle query set is not appropriate.
        """
        return self.ask(
            question=question,
            queries=sub_queries,
            verbose=True
        )


if __name__ == "__main__":

    agent = QueryAgent()

    # Run the full chanterelle habitat analysis
    print("\n" + "=" * 60)
    print("CHANTERELLE HABITAT ANALYSIS — UCLUELET BC")
    print("=" * 60 + "\n")

    result = agent.analyse_chanterelle_habitat()

    # Save the result
    os.makedirs("outputs/reports", exist_ok=True)
    report_path = "outputs/reports/chanterelle_habitat_analysis.txt"

    with open(report_path, "w") as f:
        f.write("CHANTERELLE HABITAT ANALYSIS — UCLUELET BC\n")
        f.write("=" * 60 + "\n\n")
        f.write(
            "Generated from BC government geospatial data\n"
            "via RAG pipeline over ChromaDB vector store\n\n"
        )
        f.write("=" * 60 + "\n\n")
        f.write(result)

    print(f"\nReport saved to {report_path}")

    # Follow-up questions you can run interactively
    follow_ups = [
        (
            "Which specific forest stands in the Ucluelet area "
            "have the oldest trees and least disturbance history, "
            "making them most suitable for chanterelle habitat?",
            [
                "old growth seral stage ancient forest stand age",
                "forest cover inventory species age class",
                "disturbance logging cutblock history",
                "protected area ecological reserve"
            ]
        ),
        (
            "What does the bioterrain and soil data tell us about "
            "moisture and drainage patterns across the Ucluelet "
            "peninsula that would affect fungal fruiting?",
            [
                "bioterrain soil moisture nutrient regime",
                "terrain texture surficial material drainage",
                "soil survey organic matter mineral",
                "aquifer groundwater depth water table"
            ]
        )
    ]

    print("\n" + "=" * 60)
    print("FOLLOW-UP ANALYSIS")
    print("=" * 60)

    for question, queries in follow_ups:
        agent.custom_query(question, queries)
        print()