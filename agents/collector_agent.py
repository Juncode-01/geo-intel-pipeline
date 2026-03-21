"""
Collector Agent
--------------
Responsible for hitting the BC Data Catalogue API and 
retrieving dataset metadata based on configured search terms.
Does not classify or download - just collects metadata records.
"""

import requests
import pandas as pd
import time
from config import BC_CATALOGUE_BASE_URL, SEARCH_TERMS

class CollectorAgent:
    
    def __init__(self):
        self.base_url = BC_CATALOGUE_BASE_URL
        self.collected = []
        self.seen_ids = set()  # track duplicates during collection
    
    def search(self, query, rows=50):
        """
        Hit the BC Catalogue API with a single search term.
        Returns a list of dataset metadata records.
        """
        try:
            response = requests.get(
                f"{self.base_url}/package_search",
                params={"q": query, "rows": rows},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                print(f"  API returned success=False for query: {query}")
                return []
            
            results = data["result"]["results"]
            print(f"  Query '{query}': found {data['result']['count']} total, "
                  f"retrieved {len(results)}")
            return results
        
        except requests.exceptions.Timeout:
            print(f"  Timeout on query: {query}")
            return []
        except Exception as e:
            print(f"  Error on query '{query}': {e}")
            return []
    
    def extract_metadata(self, dataset):
        """
        Pull the fields we care about from a raw API result.
        Flattens the nested structure into a simple dictionary.
        """
        # Get all resource formats and URLs
        resources = dataset.get("resources", [])
        formats = list(set([
            r.get("format", "").upper() 
            for r in resources 
            if r.get("format")
        ]))
        
        # Find WFS URL if present
        wfs_url = None
        download_url = None
        for r in resources:
            fmt = r.get("format", "").upper()
            url = r.get("url", "")
            if fmt == "WFS" or "wfs" in url.lower():
                wfs_url = url
            elif fmt in ["GEOJSON", "SHP", "GPKG", "GEOTIFF"]:
                download_url = url
        
        # Extract tags as a comma separated string
        tags = ", ".join([
            t["name"] for t in dataset.get("tags", [])
        ])
        
        return {
            "id": dataset.get("name", ""),
            "title": dataset.get("title", ""),
            "notes": dataset.get("notes", "")[:500],
            "organization": dataset.get(
                "organization", {}
            ).get("title", ""),
            "formats": ", ".join(formats),
            "tags": tags,
            "wfs_url": wfs_url or "",
            "download_url": download_url or "",
            "has_wfs": wfs_url is not None,
            "license": dataset.get("license_title", ""),
            "metadata_modified": dataset.get("metadata_modified", "")
        }
    
    def run(self, search_terms=None, rows_per_term=50):
        """
        Main method - runs all searches, deduplicates, 
        returns a clean dataframe of metadata records.
        """
        if search_terms is None:
            search_terms = SEARCH_TERMS
        
        print(f"Starting collection with {len(search_terms)} search terms...\n")
        
        for term in search_terms:
            results = self.search(term, rows=rows_per_term)
            
            for dataset in results:
                dataset_id = dataset.get("name", "")
                
                # Skip if we've already seen this dataset
                if dataset_id in self.seen_ids:
                    continue
                
                self.seen_ids.add(dataset_id)
                metadata = self.extract_metadata(dataset)
                self.collected.append(metadata)
            
            # Be polite to the API - small delay between requests
            time.sleep(0.5)
        
        df = pd.DataFrame(self.collected)
        print(f"\nCollection complete.")
        print(f"Total unique datasets collected: {len(df)}")
        print(f"Datasets with WFS endpoints: {df['has_wfs'].sum()}")
        
        return df


# Allow running this file directly to test it
if __name__ == "__main__":
    agent = CollectorAgent()
    df = agent.run()
    
    # Save raw collected metadata
    output_path = "data/raw/collected_metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("\nSample of collected data:")
    print(df[["title", "formats", "has_wfs"]].head(10))
    