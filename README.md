# geo-intel-pipeline
A pipeline for collecting, classifying, and processing BC government 
geospatial, geoclimatic, and wildlife data to feed a local LLM for predictive ecological mapping.

## Goal
Generate predictive habitat maps for species like chanterelle mushrooms
around Ucluelet BC by reasoning over soil, hydrology, vegetation, and 
climate datasets.

## Pipeline Overview
1. BC Data Catalogue API → dataset discovery
2. Random Forest classifier → relevance filtering  
3. ConverterAgent → markdown + JSONL geospatial narratives
4. ChromaDB vector store → semantic retrieval
5. Local LLM (Ollama) → prediction and map generation

## Setup
cp .env.example .env
# Fill in your API keys in .env
pip install -r requirements.txt

## Status
- [x] Manual dataset labelling (130 records)
- [ ] Random Forest classifier
- [ ] Data collection agent
- [x] Format converters
- [ ] Vector database
- [ ] Local LLM integration

## Run Full Pipeline
Run the end-to-end collection → classification → fetch → conversion flow with:

```bash
python -m pipeline.ingest
```

## Converted Output Locations
When conversion runs, files are written under `data/processed/`:
- `data/processed/text_records.jsonl` (feature-level narratives)
- `data/processed/layer_summaries.jsonl` (layer-level summaries)
- `data/processed/text_corpus/*.md` (one markdown narrative file per dataset)
