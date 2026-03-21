"""
Ingest Pipeline
---------------
Orchestrates the collection and classification of BC Catalogue datasets.
Run this file to execute the full ingestion pipeline.
"""

from agents.collector_agent import CollectorAgent
import pandas as pd

def run_ingestion():
    print("=" * 60)
    print("BC GEOSPATIAL DATA INGESTION PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1 - Collect metadata from BC Catalogue
    print("STEP 1: Collecting dataset metadata...")
    print("-" * 40)
    collector = CollectorAgent()
    df_collected = collector.run()
    
    # Save checkpoint
    df_collected.to_csv("data/raw/collected_metadata.csv", index=False)
    print(f"\nCheckpoint saved: data/raw/collected_metadata.csv")
    
    # Step 2 - Classifier comes in Step 3
    print("\nSTEP 2: Classification (coming in Step 3)")
    print("For now, reviewing collection results...\n")
    
    # Summary stats
    print("=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total unique datasets:     {len(df_collected)}")
    print(f"Datasets with WFS:         {df_collected['has_wfs'].sum()}")
    print(f"Unique organizations:      {df_collected['organization'].nunique()}")
    print(f"\nTop formats found:")
    
    # Count formats
    all_formats = []
    for fmt_str in df_collected['formats']:
        for fmt in fmt_str.split(', '):
            if fmt.strip():
                all_formats.append(fmt.strip())
    
    from collections import Counter
    for fmt, count in Counter(all_formats).most_common(8):
        print(f"  {fmt:<20} {count}")
    
    return df_collected

if __name__ == "__main__":
    df = run_ingestion()