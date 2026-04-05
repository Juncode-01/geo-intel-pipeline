"""
Converter Agent
---------------
Converts fetched GeoJSON datasets into structured text documents
suitable for LLM ingestion and vector database storage.

Each dataset produces one text document containing:
- Dataset context (title, source, feature count, coverage area)
- Schema description (what each column means)
- Feature-level descriptions (one paragraph per feature)
- Spatial summary (geographic patterns across all features)

Output: data/processed/text/<dataset_id>.txt
"""

import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime

# Input and output directories
FETCH_DIR = "data/raw/fetched"
TEXT_DIR = "data/processed/text"

# Maximum features to describe individually
# Large datasets get sampled to keep text documents manageable
MAX_FEATURES_FULL = 50
MAX_FEATURES_SAMPLE = 200

# Columns to always skip - internal IDs, geometry artifacts
SKIP_COLUMNS = {
    "gid", "objectid", "se_anno_cad_data",
    "feature_length_m", "feature_area_sqm",
    "shape", "geometry"
}


class ConverterAgent:

    def __init__(self):
        os.makedirs(TEXT_DIR, exist_ok=True)
        self.converted = []
        self.skipped = []
        self.failed = []

    # ----------------------------------------------------------
    # MAIN RUN METHOD
    # ----------------------------------------------------------

    def run(self):
        """
        Convert all fetched datasets to text.
        Skips datasets already converted.
        """
        # Find all metadata sidecars
        meta_files = sorted([
            f for f in os.listdir(FETCH_DIR)
            if f.endswith("_meta.json")
        ])

        print(f"Found {len(meta_files)} fetched datasets to convert\n")

        for i, meta_file in enumerate(meta_files):
            meta_path = os.path.join(FETCH_DIR, meta_file)

            with open(meta_path) as f:
                meta = json.load(f)

            dataset_id = meta["dataset_id"]
            title = meta["title"]
            geojson_path = meta["file_path"]

            print(f"[{i+1}/{len(meta_files)}] {title[:60]}")

            # Skip if already converted
            if self._already_converted(dataset_id):
                print(f"  Already converted, skipping.\n")
                self.skipped.append(dataset_id)
                continue

            # Skip if geojson file missing
            if not os.path.exists(geojson_path):
                print(f"  GeoJSON file not found: {geojson_path}\n")
                self.failed.append(dataset_id)
                continue

            try:
                text = self._convert_dataset(meta, geojson_path)
                self._save_text(dataset_id, text)
                char_count = len(text)
                print(f"  ✓ Converted ({char_count:,} chars)\n")
                self.converted.append(dataset_id)

            except Exception as e:
                print(f"  ✗ Failed: {e}\n")
                self.failed.append(dataset_id)

        print("=" * 60)
        print("CONVERSION SUMMARY")
        print("=" * 60)
        print(f"  Converted:  {len(self.converted)}")
        print(f"  Skipped:    {len(self.skipped)}")
        print(f"  Failed:     {len(self.failed)}")
        print(f"  Total:      {len(meta_files)}")

    # ----------------------------------------------------------
    # CORE CONVERSION
    # ----------------------------------------------------------

    def _convert_dataset(self, meta, geojson_path):
        """
        Build a structured text document from a dataset's
        metadata sidecar and its GeoJSON feature file.
        """
        gdf = gpd.read_file(geojson_path)

        sections = []

        # 1. Dataset header
        sections.append(self._build_header(meta, gdf))

        # 2. Schema description
        sections.append(self._build_schema(gdf))

        # 3. Spatial summary
        sections.append(self._build_spatial_summary(gdf))

        # 4. Feature descriptions
        sections.append(self._build_feature_descriptions(
            gdf, meta["title"]
        ))

        # 5. Ecological context hint
        # Helps the LLM connect this dataset to habitat questions
        sections.append(self._build_ecological_context(meta, gdf))

        return "\n\n".join(sections)

    # ----------------------------------------------------------
    # SECTION BUILDERS
    # ----------------------------------------------------------

    def _build_header(self, meta, gdf):
        """Dataset-level context block."""
        bounds = meta.get("bounds", {})
        fetched_at = meta.get("fetched_at", "unknown")[:10]

        lines = [
            "=== DATASET RECORD ===",
            f"Title:          {meta['title']}",
            f"Dataset ID:     {meta['dataset_id']}",
            f"Source:         {meta.get('source_type', 'unknown').upper()}",
            f"Fetched:        {fetched_at}",
            f"Feature Count:  {meta['feature_count']}",
            f"Geometry Types: {', '.join(meta.get('geometry_types', ['unknown']))}",
            f"Coverage Area:  Ucluelet, BC region",
            f"Bounding Box:   "
            f"W={bounds.get('west','?')} "
            f"S={bounds.get('south','?')} "
            f"E={bounds.get('east','?')} "
            f"N={bounds.get('north','?')}",
            f"Coordinate System: WGS84 (EPSG:4326)",
        ]

        return "\n".join(lines)

    def _build_schema(self, gdf):
        """Describe each attribute column and its value range."""
        lines = ["=== ATTRIBUTE SCHEMA ==="]

        useful_cols = [
            c for c in gdf.columns
            if c.lower() not in SKIP_COLUMNS
            and c.lower() != "geometry"
        ]

        if not useful_cols:
            lines.append("No attribute columns available.")
            return "\n".join(lines)

        for col in useful_cols:
            series = gdf[col].dropna()
            if len(series) == 0:
                continue

            dtype = gdf[col].dtype
            col_display = col.lower().replace("_", " ")

            if dtype in [np.float64, np.int64, np.float32, np.int32]:
                try:
                    mn = series.min()
                    mx = series.max()
                    mean = series.mean()
                    lines.append(
                        f"  {col_display}: numeric | "
                        f"range {mn:.2f} to {mx:.2f} | "
                        f"mean {mean:.2f}"
                    )
                except Exception:
                    lines.append(f"  {col_display}: numeric")

            else:
                unique_vals = series.astype(str).unique()
                if len(unique_vals) <= 10:
                    vals = ", ".join(sorted(unique_vals)[:10])
                    lines.append(
                        f"  {col_display}: categorical | "
                        f"values: {vals}"
                    )
                else:
                    sample = ", ".join(
                        sorted(unique_vals)[:5]
                    )
                    lines.append(
                        f"  {col_display}: text | "
                        f"{len(unique_vals)} unique values | "
                        f"sample: {sample}..."
                    )

        return "\n".join(lines)

    def _build_spatial_summary(self, gdf):
        """Summarize geographic distribution of features."""
        lines = ["=== SPATIAL SUMMARY ==="]

        total = len(gdf)
        geom_types = gdf.geometry.geom_type.value_counts()

        lines.append(f"Total features in Ucluelet area: {total}")

        for geom_type, count in geom_types.items():
            lines.append(f"  {geom_type}: {count} features")

        # For polygon data, summarize area distribution
        if "Polygon" in gdf.geometry.geom_type.values or \
           "MultiPolygon" in gdf.geometry.geom_type.values:
            try:
                # Project to BC Albers for accurate area in m²
                gdf_proj = gdf.to_crs("EPSG:3005")
                areas_ha = gdf_proj.geometry.area / 10000
                lines.append(
                    f"Area distribution (hectares): "
                    f"min={areas_ha.min():.1f} "
                    f"median={areas_ha.median():.1f} "
                    f"max={areas_ha.max():.1f}"
                )
            except Exception:
                pass

        # Centroid of all features
        try:
            centroid = gdf.geometry.union_all().centroid
            lines.append(
                f"Approximate centre: "
                f"{centroid.y:.4f}°N, {centroid.x:.4f}°W"
            )
        except Exception:
            pass

        return "\n".join(lines)

    def _build_feature_descriptions(self, gdf, dataset_title):
        """
        Generate a natural language description for each feature.
        Large datasets are sampled to keep output manageable.
        """
        lines = ["=== FEATURE DESCRIPTIONS ==="]

        total = len(gdf)

        # Decide whether to describe all or sample
        if total <= MAX_FEATURES_FULL:
            df_describe = gdf
            lines.append(
                f"Describing all {total} features:\n"
            )
        else:
            df_describe = gdf.sample(
                min(MAX_FEATURES_SAMPLE, total),
                random_state=42
            )
            lines.append(
                f"Dataset has {total} features. "
                f"Describing a representative sample "
                f"of {len(df_describe)}:\n"
            )

        useful_cols = [
            c for c in gdf.columns
            if c.lower() not in SKIP_COLUMNS
            and c.lower() != "geometry"
        ]

        for idx, row in df_describe.iterrows():
            desc = self._describe_feature(
                row, useful_cols, dataset_title, idx
            )
            lines.append(desc)

        return "\n".join(lines)

    def _describe_feature(self, row, columns, dataset_title, idx):
        """
        Convert a single feature row into a natural language
        paragraph an LLM can read and reason about.
        """
        parts = []

        # Get geometry info
        geom = row.geometry
        if geom is not None:
            try:
                centroid = geom.centroid
                location = (
                    f"located at approximately "
                    f"{centroid.y:.4f}°N, {centroid.x:.4f}°W"
                )
            except Exception:
                location = "location unavailable"
        else:
            location = "no geometry"

        parts.append(
            f"Feature {idx} from '{dataset_title}', {location}."
        )

        # Add all non-null attributes as readable key-value pairs
        attr_parts = []
        for col in columns:
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                col_readable = col.lower().replace("_", " ")
                attr_parts.append(f"{col_readable}: {val}")

        if attr_parts:
            parts.append(" | ".join(attr_parts))

        return " ".join(parts)

    def _build_ecological_context(self, meta, gdf):
        """
        Add a brief note connecting this dataset to ecological
        and habitat relevance. Helps LLM make connections when
        answering chanterelle habitat questions.
        """
        lines = ["=== ECOLOGICAL RELEVANCE ==="]

        title_lower = meta["title"].lower()
        dataset_id_lower = meta["dataset_id"].lower()
        combined = title_lower + " " + dataset_id_lower

        hints = []

        if any(w in combined for w in ["soil", "surficial", "terrain"]):
            hints.append(
                "Soil and terrain data directly informs fungal habitat "
                "suitability. Chanterelles favour well-drained but "
                "moisture-retaining soils, often sandy loam or organic "
                "mineral mixes under coniferous canopy."
            )

        if any(w in combined for w in ["water", "hydro", "watershed",
                                        "stream", "drainage", "aquifer"]):
            hints.append(
                "Hydrological data indicates soil moisture patterns and "
                "drainage regimes. Chanterelles favour areas with "
                "consistent subsurface moisture but not waterlogged "
                "conditions — typically mid-slope positions near but "
                "not in riparian zones."
            )

        if any(w in combined for w in ["forest", "vegetation", "tree",
                                        "vri", "cover", "species"]):
            hints.append(
                "Vegetation and forest cover data indicates tree species "
                "composition. Chanterelles form mycorrhizal associations "
                "primarily with conifers — especially Douglas-fir, "
                "Sitka spruce, western hemlock, and western red cedar, "
                "all common in the Ucluelet region."
            )

        if any(w in combined for w in ["elevation", "slope", "aspect",
                                        "dem", "lidar"]):
            hints.append(
                "Terrain data informs aspect and slope, which affect "
                "microclimate. North and east facing slopes in the "
                "Pacific Northwest retain more moisture and are often "
                "more productive for chanterelles than exposed "
                "south-facing slopes."
            )

        if any(w in combined for w in ["climate", "precipitation",
                                        "temperature", "weather"]):
            hints.append(
                "Climate data informs seasonal precipitation patterns. "
                "Chanterelle fruiting in coastal BC typically peaks "
                "August through November following late summer rains, "
                "with the Ucluelet area receiving some of the highest "
                "annual rainfall in Canada."
            )

        if any(w in combined for w in ["park", "protected", "conserv",
                                        "ecological reserve"]):
            hints.append(
                "Protected area data indicates zones with reduced human "
                "disturbance and intact old-growth forest — conditions "
                "highly associated with productive chanterelle habitat."
            )

        if not hints:
            hints.append(
                "This dataset may provide supporting spatial context "
                "for habitat analysis in the Ucluelet region."
            )

        lines.extend(hints)
        return "\n".join(lines)

    # ----------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------

    def _already_converted(self, dataset_id):
        """Check if text output already exists for this dataset."""
        safe_id = dataset_id.replace(
            "/", "_"
        ).replace(" ", "_").replace(":", "_")
        path = os.path.join(TEXT_DIR, f"{safe_id}.txt")
        return os.path.exists(path)

    def _save_text(self, dataset_id, text):
        """Save the converted text document."""
        safe_id = dataset_id.replace(
            "/", "_"
        ).replace(" ", "_").replace(":", "_")
        path = os.path.join(TEXT_DIR, f"{safe_id}.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    agent = ConverterAgent()
    agent.run()

    # Show a sample of what was produced
    import os
    files = [
        f for f in os.listdir(TEXT_DIR)
        if f.endswith(".txt")
    ]
    if files:
        sample_path = os.path.join(TEXT_DIR, files[0])
        print(f"\n--- Sample output: {files[0]} ---\n")
        with open(sample_path) as f:
            print(f.read()[:2000])