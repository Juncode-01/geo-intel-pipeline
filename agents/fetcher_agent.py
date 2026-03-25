"""
Fetcher Agent
-------------
Takes classified relevant datasets and fetches actual geographic data.
Handles WFS services, GeoJSON downloads, and Shapefiles.
Clips all results to the Ucluelet bounding box.
Saves standardized GeoJSON files to data/raw/fetched/

Each saved file is accompanied by a metadata sidecar JSON
describing what it contains, where it came from, and when it was fetched.
"""

import os
import json
import time
import zipfile
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from datetime import datetime
from io import BytesIO
from fetcher.wfs_client import get_capabilities, fetch_layer

from config import (
    UCLUELET_BBOX,
    MAX_WFS_FEATURES,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    MAX_DOWNLOAD_MB,
    WFS_FORMAT_PREFERENCE
)

# Output directory for fetched data
FETCH_OUTPUT_DIR = "data/raw/fetched"


class FetcherAgent:

    def __init__(self):
        os.makedirs(FETCH_OUTPUT_DIR, exist_ok=True)

        # Build a shapely box for spatial clipping
        self.bbox = UCLUELET_BBOX
        self.clip_box = box(
            self.bbox[0],  # min lon (west)
            self.bbox[1],  # min lat (south)
            self.bbox[2],  # max lon (east)
            self.bbox[3]   # max lat (north)
        )

        # Track results
        self.results = []

    # ----------------------------------------------------------
    # MAIN RUN METHOD
    # ----------------------------------------------------------

    def run(self, df_relevant, min_score=0.6):
        """
        Process all relevant datasets.
        Tries WFS first (cleanest), then download fallback.
        """
        # Filter to datasets above score threshold
        df = df_relevant[
            df_relevant["relevance_score"] >= min_score
        ].copy()

        print(f"Fetching data for {len(df)} relevant datasets...")
        print(f"Bounding box: {self.bbox}\n")

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, row in df.iterrows():
            dataset_id = row["id"]
            title = row["title"]
            score = row["relevance_score"]

            print(f"[{i+1}/{len(df)}] {title[:60]}")
            print(f"         Score: {score:.2f}")

            # Skip if already fetched
            if self._already_fetched(dataset_id):
                print(f"         Already fetched, skipping.\n")
                skip_count += 1
                continue

            result = None

            # Try WFS first - cleanest and most targeted
            if row.get("has_wfs") and row.get("wfs_url"):
                print(f"         Trying WFS...")
                result = self._fetch_wfs(
                    dataset_id, title, row["wfs_url"]
                )

            # Fall back to direct download
            if result is None and row.get("download_url"):
                print(f"         Trying direct download...")
                result = self._fetch_download(
                    dataset_id, title, row["download_url"]
                )

            if result and result["success"]:
                print(f"         ✓ Fetched {result['feature_count']} features\n")
                success_count += 1
                self.results.append(result)
            else:
                reason = result["reason"] if result else "no available endpoint"
                print(f"         ✗ Failed: {reason}\n")
                fail_count += 1

            # Polite delay between requests
            time.sleep(1)

        print("=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)
        print(f"  Successful:  {success_count}")
        print(f"  Skipped:     {skip_count}")
        print(f"  Failed:      {fail_count}")
        print(f"  Total:       {len(df)}")
        print()

        return self.results

    # ----------------------------------------------------------
    # WFS FETCHING
    # ----------------------------------------------------------

    def _fetch_wfs(self, dataset_id, title, wfs_url):
        clean_id = dataset_id.removeprefix("pub:")
        """
        Fetch features from a WFS endpoint clipped to bounding box.
        Tries multiple output formats until one works.
        """
        bbox_str = (
            f"{self.bbox[0]},{self.bbox[1]},"
            f"{self.bbox[2]},{self.bbox[3]},"
            f"EPSG:4326"
        )

        for fmt in WFS_FORMAT_PREFERENCE:
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeName": clean_id,
                "outputFormat": fmt,
                "count": MAX_WFS_FEATURES,
                "bbox": bbox_str
            }

            try:
                response = self._request_with_retry(wfs_url, params)
                if response is None:
                    continue

                # Check content type
                content_type = response.headers.get(
                    "content-type", ""
                ).lower()

                if "json" in content_type or "geo" in content_type:
                    try:
                        geojson = response.json()
                        features = geojson.get("features", [])

                        if len(features) == 0:
                            # No features in this area - valid but empty
                            return {
                                "success": False,
                                "reason": "no features in bounding box"
                            }

                        # Convert to GeoDataFrame and clip
                        gdf = gpd.GeoDataFrame.from_features(features)
                        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
                        gdf_clipped = self._clip_to_bbox(gdf)

                        if len(gdf_clipped) == 0:
                            return {
                                "success": False,
                                "reason": "no features after clipping"
                            }

                        return self._save_result(
                            dataset_id, title, gdf_clipped, "wfs"
                        )

                    except Exception as e:
                        print(f"         JSON parse error: {e}")
                        continue

            except Exception as e:
                print(f"         WFS error ({fmt}): {e}")
                continue

        return {"success": False, "reason": "all WFS formats failed"}

    # ----------------------------------------------------------
    # DOWNLOAD FETCHING
    # ----------------------------------------------------------

    def _fetch_download(self, dataset_id, title, download_url):
        """
        Download a file (GeoJSON, Shapefile zip, GeoPackage)
        and clip to bounding box.
        """
        try:
            # Check file size with HEAD request first
            head = requests.head(
                download_url, timeout=10, allow_redirects=True
            )
            content_length = int(
                head.headers.get("content-length", 0)
            )
            size_mb = content_length / (1024 * 1024)

            if size_mb > MAX_DOWNLOAD_MB:
                return {
                    "success": False,
                    "reason": f"file too large ({size_mb:.0f}MB > {MAX_DOWNLOAD_MB}MB)"
                }

            # Download the file
            response = self._request_with_retry(download_url, {})
            if response is None:
                return {"success": False, "reason": "download failed"}

            url_lower = download_url.lower()

            # Route by file type
            if url_lower.endswith(".geojson") or url_lower.endswith(".json"):
                gdf = self._load_geojson(response.content)

            elif url_lower.endswith(".zip"):
                gdf = self._load_shapefile_zip(response.content)

            elif url_lower.endswith(".gpkg"):
                gdf = self._load_geopackage(response.content)

            else:
                return {
                    "success": False,
                    "reason": f"unsupported format: {url_lower.split('.')[-1]}"
                }

            if gdf is None or len(gdf) == 0:
                return {"success": False, "reason": "empty dataset"}

            # Reproject to WGS84 if needed
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            # Clip to bounding box
            gdf_clipped = self._clip_to_bbox(gdf)

            if len(gdf_clipped) == 0:
                return {
                    "success": False,
                    "reason": "no features in Ucluelet area"
                }

            return self._save_result(
                dataset_id, title, gdf_clipped, "download"
            )

        except Exception as e:
            return {"success": False, "reason": str(e)}

    # ----------------------------------------------------------
    # FILE LOADERS
    # ----------------------------------------------------------

    def _load_geojson(self, content):
        """Load GeoJSON bytes into a GeoDataFrame."""
        try:
            return gpd.read_file(BytesIO(content))
        except Exception as e:
            print(f"         GeoJSON load error: {e}")
            return None

    def _load_shapefile_zip(self, content):
        """
        Unzip a shapefile archive and load it.
        Shapefiles come as zip archives containing
        multiple component files (.shp, .dbf, .prj etc.)
        """
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                # Find the .shp file inside the zip
                shp_files = [
                    f for f in zf.namelist()
                    if f.endswith(".shp")
                ]

                if not shp_files:
                    print("         No .shp file found in zip")
                    return None

                # Extract to a temp location
                extract_path = "data/raw/temp_shp"
                os.makedirs(extract_path, exist_ok=True)
                zf.extractall(extract_path)

                shp_path = os.path.join(
                    extract_path, shp_files[0]
                )
                return gpd.read_file(shp_path)

        except Exception as e:
            print(f"         Shapefile load error: {e}")
            return None

    def _load_geopackage(self, content):
        """Load a GeoPackage file."""
        try:
            # Save temporarily then read
            temp_path = "data/raw/temp.gpkg"
            with open(temp_path, "wb") as f:
                f.write(content)
            return gpd.read_file(temp_path)
        except Exception as e:
            print(f"         GeoPackage load error: {e}")
            return None

    # ----------------------------------------------------------
    # SPATIAL UTILITIES
    # ----------------------------------------------------------

    def _clip_to_bbox(self, gdf):
        """Clip a GeoDataFrame to the Ucluelet bounding box."""
        try:
            return gdf.clip(self.clip_box)
        except Exception as e:
            # Fallback: filter by centroid if clip fails
            try:
                centroids = gdf.geometry.centroid
                mask = (
                    (centroids.x >= self.bbox[0]) &
                    (centroids.x <= self.bbox[2]) &
                    (centroids.y >= self.bbox[1]) &
                    (centroids.y <= self.bbox[3])
                )
                return gdf[mask]
            except Exception:
                return gdf  # return unclipped if all else fails

    # ----------------------------------------------------------
    # SAVING
    # ----------------------------------------------------------

    def _save_result(self, dataset_id, title, gdf, source_type):
        """
        Save a clipped GeoDataFrame as GeoJSON with a
        metadata sidecar file describing its contents.
        """
        # Clean dataset_id for use as filename
        safe_id = dataset_id.replace("/", "_").replace(" ", "_")
        geojson_path = os.path.join(
            FETCH_OUTPUT_DIR, f"{safe_id}.geojson"
        )
        sidecar_path = os.path.join(
            FETCH_OUTPUT_DIR, f"{safe_id}_meta.json"
        )

        # Save GeoJSON
        gdf.to_file(geojson_path, driver="GeoJSON")

        # Build metadata sidecar
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        columns = [
            c for c in gdf.columns
            if c != "geometry"
        ]

        metadata = {
            "dataset_id": dataset_id,
            "title": title,
            "source_type": source_type,
            "fetched_at": datetime.now().isoformat(),
            "feature_count": len(gdf),
            "geometry_types": list(
                gdf.geometry.geom_type.unique()
            ),
            "bounds": {
                "west": round(bounds[0], 6),
                "south": round(bounds[1], 6),
                "east": round(bounds[2], 6),
                "north": round(bounds[3], 6)
            },
            "attribute_columns": columns,
            "crs": "EPSG:4326",
            "file_path": geojson_path
        }

        with open(sidecar_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "title": title,
            "feature_count": len(gdf),
            "file_path": geojson_path,
            "metadata_path": sidecar_path,
            "reason": None
        }

    # ----------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------

    def _already_fetched(self, dataset_id):
        """Check if we already have data for this dataset."""
        safe_id = dataset_id.replace("/", "_").replace(" ", "_")
        path = os.path.join(FETCH_OUTPUT_DIR, f"{safe_id}.geojson")
        return os.path.exists(path)

    def _request_with_retry(self, url, params):
        """
        Make an HTTP request with automatic retry on failure.
        Waits progressively longer between retries.
        """
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    stream=False
                )
                response.raise_for_status()
                return response

            except requests.exceptions.Timeout:
                wait = (attempt + 1) * 5
                print(f"         Timeout (attempt {attempt+1}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)

            except requests.exceptions.HTTPError as e:
                print(f"         HTTP error: {e}")
                return None

            except Exception as e:
                print(f"         Request error: {e}")
                return None

        return None

    def get_fetch_summary(self):
        """Print a summary of everything fetched so far."""
        files = [
            f for f in os.listdir(FETCH_OUTPUT_DIR)
            if f.endswith("_meta.json")
        ]

        if not files:
            print("Nothing fetched yet.")
            return

        print(f"\nFetched datasets in {FETCH_OUTPUT_DIR}:")
        print("-" * 60)
        total_features = 0
        for f in sorted(files):
            path = os.path.join(FETCH_OUTPUT_DIR, f)
            with open(path) as fp:
                meta = json.load(fp)
            total_features += meta["feature_count"]
            print(f"  {meta['title'][:50]}")
            print(f"    Features: {meta['feature_count']} | "
                  f"Type: {meta['source_type']} | "
                  f"Columns: {len(meta['attribute_columns'])}")

        print(f"\nTotal features across all datasets: {total_features}")

    def discover_wfs_layers(self, keywords=None):
        """
        Pull all available layers from BC WFS and return them
        as a DataFrame in the same shape as relevant_datasets.csv
        so they can be passed straight into run().
        """
        print("Discovering layers from BC WFS...")
        layer_names = get_capabilities()
        print(f"Found {len(layer_names)} available layers\n")
    
        rows = []
        for name in layer_names:
            # Skip if already fetched
            if self._already_fetched(name):
                continue
    
            # Optional keyword filter so you don't fetch everything blindly
            if keywords:
                name_lower = name.lower()
                if not any(kw.lower() in name_lower for kw in keywords):
                    continue
    
            rows.append({
                "id": name,
                "title": name,
                "relevance_score": 1.0,   # bypass RF threshold for now
                "has_wfs": True,
                "wfs_url": "https://openmaps.gov.bc.ca/geo/pub/wfs",
                "download_url": None
            })
    
        df = pd.DataFrame(rows)
        print(f"Queuing {len(df)} layers for fetching\n")
        return df


if __name__ == "__main__":
    agent = FetcherAgent()

    # --- Option A: use your existing classified CSV (geology only) ---
    # try:
    #     df_relevant = pd.read_csv("data/processed/relevant_datasets.csv")
    # except FileNotFoundError:
    #     print("relevant_datasets.csv not found.")
    #     exit(1)

    # --- Option B: discover directly from WFS (use this now) ---
    df_relevant = agent.discover_wfs_layers(keywords=[
        "water", "watershed", "aquifer",
        "road", "forest", "soil",
        "wildlife", "fish", "marine",
        "shoreline", "coastal", "riparian",
        "zoning", "administrative", "park",
        "terrain", "slope", "elevation"
    ])

    if df_relevant.empty:
        print("No new layers found to fetch.")
        exit(0)

    results = agent.run(df_relevant, min_score=0.6)

    print("\nFetch complete. Summary:")
    agent.get_fetch_summary()
