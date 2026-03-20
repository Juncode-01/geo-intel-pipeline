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
3. Format converters → structured text/JSON
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
- [ ] Format converters
- [ ] Vector database
- [ ] Local LLM integration
