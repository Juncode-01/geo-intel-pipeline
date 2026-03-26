# Converter Agent

Converts fetched vector geospatial data in `data/raw/fetched/` into structured text artifacts that are ready for local-LLM training and RAG ingestion.

## Inputs
- `*.geojson` datasets fetched by `FetcherAgent`
- `*_meta.json` sidecars generated during fetch

## Outputs
- `data/processed/text_corpus/<dataset_id>.md`: human-readable layer narratives + feature bullets
- `data/processed/layer_summaries.jsonl`: one summary record per layer
- `data/processed/text_records.jsonl`: one narrative record per feature

## Behavior
- Reads each layer sidecar and corresponding GeoJSON.
- Builds layer-level stats:
  - geometry types and bounds
  - numeric column summaries (mean/min/max/std)
  - top categorical values
- Builds per-feature narratives with centroid + key attributes.
- Writes both markdown and JSONL outputs for easy indexing in a vector DB.

Run directly:

```bash
python agents/converter_agent.py/converter.py
```
