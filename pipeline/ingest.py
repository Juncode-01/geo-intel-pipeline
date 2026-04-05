"""
Ingest Pipeline
---------------
Orchestrates collection, classification, and fetching
of BC Catalogue geospatial datasets for the Ucluelet area.
"""

import pandas as pd
from agents.collector_agent import CollectorAgent
from agents.classifier_agent import ClassifierAgent
from agents.fetcher_agent import FetcherAgent
from agents.converter_agent import ConverterAgent
from agents.vector_store_agent import VectorStoreAgent
from agents.query_agent import QueryAgent
from config import WFS_DISCOVERY_KEYWORDS


def run_ingestion(
    score_threshold=0.6,
    fetch_data=True,
    include_wfs_discovery=True,
    wfs_keywords=None
):

    print("=" * 60)
    print("BC GEOSPATIAL DATA INGESTION PIPELINE")
    print("=" * 60)
    print()

    # --- Step 1: Collect ---
    print("STEP 1: Collecting dataset metadata...")
    print("-" * 40)
    collector = CollectorAgent()
    df_collected = collector.run()
    df_collected.to_csv(
        "data/raw/collected_metadata.csv", index=False
    )
    print(f"Saved: data/raw/collected_metadata.csv\n")

    # --- Step 2: Classify ---
    print("STEP 2: Classifying for relevance...")
    print("-" * 40)
    classifier = ClassifierAgent()

    if not classifier._load_model():
        print("No saved model found. Training now...")
        classifier.train()

    df_scored = classifier.predict(df_collected)
    df_scored.to_csv(
        "data/processed/all_scored.csv", index=False
    )

    df_relevant = df_scored[
        df_scored["relevance_score"] >= score_threshold
    ].copy()
    df_relevant.to_csv(
        "data/processed/relevant_datasets.csv", index=False
    )

    print(f"\nRelevant datasets: {len(df_relevant)}")
    print(f"Top 5:")
    for _, row in df_relevant.head(5).iterrows():
        print(f"  [{row['relevance_score']:.2f}] {row['title']}")
    print()

    # --- Step 3: Fetch ---
    if fetch_data:
        print("STEP 3: Fetching geographic data...")
        print("-" * 40)
        fetcher = FetcherAgent()

        df_fetch_targets = df_relevant.copy()

        # Optionally merge in direct WFS discovery layers from BC OpenMaps.
        # These rows are shaped to match relevant_datasets.csv and can be
        # fetched by the same FetcherAgent.run() method.
        if include_wfs_discovery:
            keywords = wfs_keywords or WFS_DISCOVERY_KEYWORDS
            df_wfs = fetcher.discover_wfs_layers(keywords=keywords)

            if not df_wfs.empty:
                df_fetch_targets = pd.concat(
                    [df_fetch_targets, df_wfs],
                    ignore_index=True
                )
                # De-duplicate by id so the same layer is not fetched twice.
                # Keep first row (catalogue-scored rows come first).
                df_fetch_targets = df_fetch_targets.drop_duplicates(
                    subset=["id"], keep="first"
                ).reset_index(drop=True)

                print(
                    "Merged fetch targets: "
                    f"{len(df_relevant)} classified + "
                    f"{len(df_wfs)} WFS-discovered -> "
                    f"{len(df_fetch_targets)} unique datasets\n"
                )

        fetcher.run(df_fetch_targets, min_score=score_threshold)
        fetcher.get_fetch_summary()

    # --- Step 4: Convert fetched data to LLM-readable text ---
    print("STEP 4: Converting geographic data to text...")
    print("-" * 40)
    converter = ConverterAgent()
    converter.run()

    # --- Step 5: Ingest text into ChromaDB vector store ---
    print("STEP 5: Ingesting text documents into ChromaDB...")
    print("-" * 40)
    vector_store = VectorStoreAgent()
    vector_store.ingest_all()
    vector_store.get_stats()

    print("\nIngestion pipeline complete.")

    return df_relevant


if __name__ == "__main__":
    run_ingestion(score_threshold=0.6, fetch_data=True)
