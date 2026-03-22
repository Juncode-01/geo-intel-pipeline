"""
Ingest Pipeline
---------------
Orchestrates collection and classification of BC Catalogue datasets.
"""

import pandas as pd
from agents.collector_agent import CollectorAgent
from agents.classifier_agent import ClassifierAgent


def run_ingestion(score_threshold=0.6):
    """
    score_threshold: minimum relevance_score to keep a dataset.
    Lower = more permissive, higher = stricter filtering.
    Start at 0.6 and adjust based on results.
    """
    print("=" * 60)
    print("BC GEOSPATIAL DATA INGESTION PIPELINE")
    print("=" * 60)
    print()

    # --- Step 1: Collect ---
    print("STEP 1: Collecting dataset metadata from BC Catalogue...")
    print("-" * 40)
    collector = CollectorAgent()
    df_collected = collector.run()
    df_collected.to_csv("data/raw/collected_metadata.csv", index=False)
    print(f"Saved raw collection: data/raw/collected_metadata.csv\n")

    # --- Step 2: Classify ---
    print("STEP 2: Classifying datasets for relevance...")
    print("-" * 40)
    classifier = ClassifierAgent()

    # Try loading existing model first, train if none exists
    if not classifier._load_model():
        print("No saved model found. Training now...")
        classifier.train()

    df_scored = classifier.predict(df_collected)

    # Save full scored results
    df_scored.to_csv("data/processed/all_scored.csv", index=False)
    print(f"Saved scored results: data/processed/all_scored.csv")

    # Filter to relevant only
    df_relevant = df_scored[
        df_scored["relevance_score"] >= score_threshold
    ].copy()

    df_relevant.to_csv("data/processed/relevant_datasets.csv", index=False)

    # --- Summary ---
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total collected:          {len(df_collected)}")
    print(f"Predicted relevant:       {len(df_relevant)}")
    print(f"Relevance threshold used: {score_threshold}")
    print(f"Datasets with WFS:        {df_relevant['has_wfs'].sum()}")
    print()
    print("Top 10 most relevant datasets:")
    print("-" * 40)
    for _, row in df_relevant.head(10).iterrows():
        print(f"  [{row['relevance_score']:.2f}] {row['title']}")
    print()
    print(f"Relevant datasets saved: data/processed/relevant_datasets.csv")

    return df_relevant


if __name__ == "__main__":
    df = run_ingestion(score_threshold=0.6)