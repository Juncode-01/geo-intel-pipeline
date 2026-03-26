"""
Converter Agent
---------------
Converts fetched geospatial layers in data/raw/fetched into
structured text artifacts optimized for local LLM training and RAG.

Outputs:
- data/processed/text_corpus/<dataset_id>.md
- data/processed/text_records.jsonl (one record per feature)
- data/processed/layer_summaries.jsonl (one summary per layer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd

FETCH_DIR = Path("data/raw/fetched")
TEXT_DIR = Path("data/processed/text_corpus")
FEATURE_JSONL = Path("data/processed/text_records.jsonl")
LAYER_JSONL = Path("data/processed/layer_summaries.jsonl")


@dataclass
class LayerConversionResult:
    dataset_id: str
    title: str
    feature_count: int
    text_path: str
    skipped: bool = False
    reason: str | None = None


class ConverterAgent:
    """Build rich, geospatially-aware text corpora from fetched vector files."""

    def __init__(self, fetched_dir: Path = FETCH_DIR):
        self.fetched_dir = Path(fetched_dir)
        TEXT_DIR.mkdir(parents=True, exist_ok=True)
        FEATURE_JSONL.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[LayerConversionResult] = []

    def run(self) -> List[LayerConversionResult]:
        sidecars = sorted(self.fetched_dir.glob("*_meta.json"))
        if not sidecars:
            print("No fetched metadata sidecars found in data/raw/fetched.")
            return []

        self._reset_output_files()

        for sidecar in sidecars:
            result = self._convert_layer(sidecar)
            self.results.append(result)

        self._print_summary()
        return self.results

    def _reset_output_files(self) -> None:
        for file_path in (FEATURE_JSONL, LAYER_JSONL):
            if file_path.exists():
                file_path.unlink()

    def _convert_layer(self, sidecar_path: Path) -> LayerConversionResult:
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))

        dataset_id = meta.get("dataset_id", sidecar_path.stem.replace("_meta", ""))
        title = meta.get("title", dataset_id)
        geojson_path = Path(meta.get("file_path", ""))

        if not geojson_path.exists():
            return LayerConversionResult(dataset_id, title, 0, "", True, f"GeoJSON not found: {geojson_path}")

        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            return LayerConversionResult(dataset_id, title, 0, "", True, "Layer is empty")

        if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        text_path = TEXT_DIR / f"{self._safe_id(dataset_id)}.md"

        layer_context = self._build_layer_context(meta, gdf)
        feature_records = self._build_feature_records(meta, gdf)

        self._write_markdown(text_path, layer_context, feature_records)
        self._append_jsonl(LAYER_JSONL, layer_context)
        for record in feature_records:
            self._append_jsonl(FEATURE_JSONL, record)

        return LayerConversionResult(dataset_id, title, len(gdf), str(text_path))

    def _build_layer_context(self, meta: Dict, gdf: gpd.GeoDataFrame) -> Dict:
        numeric_stats = self._numeric_stats(gdf)
        top_categories = self._top_categories(gdf)

        return {
            "record_type": "layer_summary",
            "dataset_id": meta.get("dataset_id"),
            "title": meta.get("title"),
            "source_type": meta.get("source_type"),
            "fetched_at": meta.get("fetched_at"),
            "feature_count": int(len(gdf)),
            "geometry_types": sorted(set(gdf.geometry.geom_type.astype(str))),
            "bounds": meta.get("bounds", {}),
            "attribute_columns": [c for c in gdf.columns if c != "geometry"],
            "numeric_stats": numeric_stats,
            "top_categories": top_categories,
            "text": self._compose_layer_description(meta, gdf, numeric_stats, top_categories),
            "created_at": datetime.utcnow().isoformat(),
        }

    def _build_feature_records(self, meta: Dict, gdf: gpd.GeoDataFrame) -> List[Dict]:
        records: List[Dict] = []
        for idx, row in gdf.iterrows():
            attrs = self._clean_attributes(row.drop(labels=["geometry"]).to_dict())
            centroid = row.geometry.centroid if row.geometry is not None else None
            centroid_repr = None
            if centroid is not None and not centroid.is_empty:
                centroid_repr = {"lon": round(float(centroid.x), 6), "lat": round(float(centroid.y), 6)}

            records.append(
                {
                    "record_type": "feature",
                    "dataset_id": meta.get("dataset_id"),
                    "dataset_title": meta.get("title"),
                    "feature_index": int(idx),
                    "geometry_type": str(row.geometry.geom_type) if row.geometry is not None else None,
                    "centroid": centroid_repr,
                    "attributes": attrs,
                    "text": self._compose_feature_description(
                        dataset_title=meta.get("title", ""),
                        dataset_id=meta.get("dataset_id", ""),
                        row_index=int(idx),
                        geometry_type=str(row.geometry.geom_type) if row.geometry is not None else "unknown",
                        attributes=attrs,
                        centroid=centroid_repr,
                    ),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
        return records

    def _compose_layer_description(self, meta: Dict, gdf: gpd.GeoDataFrame, numeric_stats: Dict, top_categories: Dict) -> str:
        lines = [
            f"Layer '{meta.get('title')}' ({meta.get('dataset_id')}) contains {len(gdf)} features.",
            f"Geometry types observed: {', '.join(sorted(set(gdf.geometry.geom_type.astype(str))))}.",
        ]
        bounds = meta.get("bounds") or {}
        if bounds:
            lines.append(
                "Spatial extent (WGS84): west {west}, south {south}, east {east}, north {north}.".format(
                    west=bounds.get("west"),
                    south=bounds.get("south"),
                    east=bounds.get("east"),
                    north=bounds.get("north"),
                )
            )
        if numeric_stats:
            joined = "; ".join(
                f"{col}: mean={vals['mean']}, min={vals['min']}, max={vals['max']}"
                for col, vals in numeric_stats.items()
            )
            lines.append(f"Numeric profile: {joined}.")
        if top_categories:
            cat_bits = []
            for col, values in top_categories.items():
                rendered = ", ".join(f"{v['value']} ({v['count']})" for v in values)
                cat_bits.append(f"{col}: {rendered}")
            lines.append(f"Categorical profile: {'; '.join(cat_bits)}.")

        lines.append("Use this layer to reason about local environmental conditions and their spatial distribution.")
        return " ".join(lines)

    def _compose_feature_description(self, dataset_title: str, dataset_id: str, row_index: int, geometry_type: str, attributes: Dict, centroid: Dict | None) -> str:
        attr_text = ", ".join(f"{k}={v}" for k, v in attributes.items()) if attributes else "no attributes"
        location_text = f"centroid at lon {centroid['lon']}, lat {centroid['lat']}" if centroid else "centroid unavailable"
        return (
            f"Feature {row_index} from layer '{dataset_title}' ({dataset_id}) is a {geometry_type} with "
            f"{location_text}. Key properties: {attr_text}."
        )

    def _write_markdown(self, path: Path, layer_context: Dict, feature_records: List[Dict]) -> None:
        lines = [
            f"# {layer_context['title']}",
            "",
            "## Layer Summary",
            layer_context["text"],
            "",
            "## Feature Narratives",
            "",
        ]
        for rec in feature_records:
            lines.append(f"- {rec['text']}")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _numeric_stats(self, gdf: gpd.GeoDataFrame) -> Dict:
        numeric_cols = [c for c in gdf.columns if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]
        stats: Dict[str, Dict] = {}
        for col in numeric_cols:
            series = gdf[col].dropna()
            if series.empty:
                continue
            stats[col] = {
                "mean": self._round_float(series.mean()),
                "min": self._round_float(series.min()),
                "max": self._round_float(series.max()),
                "std": self._round_float(series.std()) if len(series) > 1 else 0.0,
            }
        return stats

    def _top_categories(self, gdf: gpd.GeoDataFrame, max_columns: int = 5, max_values: int = 5) -> Dict:
        object_cols = [
            c for c in gdf.columns
            if c != "geometry" and (pd.api.types.is_object_dtype(gdf[c]) or pd.api.types.is_categorical_dtype(gdf[c]))
        ]
        top: Dict[str, List[Dict]] = {}
        for col in object_cols[:max_columns]:
            vc = gdf[col].dropna().astype(str).value_counts().head(max_values)
            if vc.empty:
                continue
            top[col] = [{"value": idx, "count": int(cnt)} for idx, cnt in vc.items()]
        return top

    def _clean_attributes(self, attrs: Dict) -> Dict:
        clean = {}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (np.floating, float)):
                clean[key] = self._round_float(value)
            elif isinstance(value, (np.integer, int)):
                clean[key] = int(value)
            elif isinstance(value, (pd.Timestamp, datetime)):
                clean[key] = value.isoformat()
            else:
                text = str(value).strip()
                if text:
                    clean[key] = text
        return clean

    @staticmethod
    def _append_jsonl(output_path: Path, record: Dict) -> None:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_id(dataset_id: str) -> str:
        return dataset_id.replace("/", "_").replace(" ", "_")

    @staticmethod
    def _round_float(value: float, digits: int = 4) -> float:
        return round(float(value), digits)

    def _print_summary(self) -> None:
        converted = [r for r in self.results if not r.skipped]
        skipped = [r for r in self.results if r.skipped]
        print("=" * 60)
        print("CONVERTER SUMMARY")
        print("=" * 60)
        print(f"Converted layers: {len(converted)}")
        print(f"Skipped layers:   {len(skipped)}")
        if skipped:
            for result in skipped[:10]:
                print(f"- {result.dataset_id}: {result.reason}")


if __name__ == "__main__":
    ConverterAgent().run()
