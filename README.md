# geo-intel-pipeline

A modular geospatial ML pipeline that ingests BC government datasets,
classifies their ecological relevance using a trained Random Forest
classifier, converts raw vector data into structured LLM-readable text,
and generates ecologically grounded habitat predictions via a RAG-based
query system backed by a ChromaDB vector store.

**Current proof of concept:** Chanterelle mushroom habitat prediction
around Ucluelet, BC (bounding box: -125.9, 48.8, -125.4, 49.0)

---

## Pipeline Architecture
```
BC Data Catalogue API
        ↓
  CollectorAgent          — discovers datasets via CKAN API
                            (no key required)
        ↓
RF Relevance Classifier   — scores datasets using a trained
                            Random Forest
        ↓
  FetcherAgent            — pulls WFS features clipped to
                            bounding box
        ↓
  ConverterAgent          — transforms GeoJSON into structured
                            LLM-readable text
        ↓
ChromaDB Vector Store     — chunks and embeds converted text
                            (all-MiniLM-L6-v2)
        ↓
   QueryAgent             — multi-query RAG over vector store
                            via Gemini 1.5 Flash
```
---

## Data Sources

All data is sourced from the
[BC Data Catalogue](https://catalogue.data.gov.bc.ca/) via the public
WFS endpoint (`https://openmaps.gov.bc.ca/geo/pub/wfs`). No API key
is required.

The pipeline currently ingests 180+ datasets and 16,000+ features
across the following BC government namespaces:

- `WHSE_TERRESTRIAL_ECOLOGY` — soil surveys, bioterrain, broad ecosystem inventory
- `WHSE_FOREST_VEGETATION` — biogeoclimatic zones (BEC), vegetation cover, seral stage
- `WHSE_WATER_MANAGEMENT` — aquifers, hydrologic zones, watershed boundaries
- `WHSE_BASEMAPPING` — freshwater atlas, watercourses, elevation points
- `WHSE_FISH` — fish observations, watershed polygons, lake surveys
- `WHSE_WILDLIFE_MANAGEMENT` — critical habitat, wildlife habitat areas, coastal resource inventory
- `WHSE_FOREST_TENURE` — cutblocks, harvest authorities, road networks
- `WHSE_WILDLIFE_INVENTORY` — species observation points

---

## Key Technical Notes

**CRS:** BC WFS returns coordinates in BC Albers (EPSG:3005) by default.
The fetcher explicitly requests `srsName=EPSG:4326` to ensure correct
bounding box clipping.

**WFS vs WMS:** The BC Data Catalogue predominantly advertises WMS
(view-only) endpoints. WFS (data-accessible) layers are discovered
directly via the BC WFS endpoint rather than catalogue metadata.

**LLM text conversion:** Each dataset is converted to a structured text
block containing a dataset header, attribute schema, spatial summary,
and per-feature descriptions with ecological relevance context — making
raw vector data interpretable by the LLM reasoning layer.

**Query strategy:** The QueryAgent decomposes ecological queries into
constituent signals (soil moisture, tree species, terrain, hydrology)
rather than querying topic names directly, which significantly improves
retrieval quality.

**Data gaps:** The pipeline explicitly flags missing or sparse data
layers in its output rather than filling gaps with assumptions.

---

## Repository Structure
agents/           — CollectorAgent, FetcherAgent, ConverterAgent,
ClassifierAgent, VectorStoreAgent, QueryAgent
pipeline/         — ingest.py, process.py, query.py, ask.py
fetcher/          — WFS client (OWSLib wrapper)
models/           — trained Random Forest classifier (.pkl)
data/
labels/         — manually labelled training data (130 records)
raw/fetched/    — raw GeoJSON + metadata per dataset
processed/text/ — LLM-readable converted text files
chromadb/       — vector store
notebooks/        — BC Data Catalogue API exploration
outputs/reports/  — chanterelle habitat analysis output
config.py         — bounding box, model settings, paths

---

## Setup
```bash
cp .env.example .env
# Add your Gemini API key to .env

pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python -m pipeline.ingest    # collect + fetch + convert
python -m pipeline.process   # classify + embed into ChromaDB
python -m pipeline.query     # run habitat query
```

> Note: Use `python -m pipeline.ingest` (not `python pipeline/ingest.py`)
> for correct module path resolution.

---

## Classifier

The Random Forest relevance classifier is trained on 130 manually
labelled BC Data Catalogue records. Labels reflect ecological
relevance to habitat prediction tasks. Current F1: ~0.667.
Additional training data from the 180+ fetched layers is being
incorporated to improve performance toward a 0.7 target.

---

## Status

- [x] BC Data Catalogue API integration (CKAN)
- [x] WFS feature fetching with bbox clipping (EPSG:4326)
- [x] Random Forest relevance classifier
- [x] GeoJSON → structured LLM-readable text conversion
- [x] ChromaDB vector store with sentence-transformer embeddings
- [x] Multi-query RAG via Gemini 1.5 Flash
- [x] Chanterelle habitat analysis output (Ucluelet, BC)
- [ ] Improved classifier F1 (target: 0.70+)
- [ ] Field validation of habitat predictions
- [ ] Pipeline generalization to additional species/ecological tasks
