# Process Pipeline

Orchestrates conversion from fetched geospatial files to LLM/RAG-friendly text.

Run:

```bash
python pipeline/process.py/convert_fetched.py
```

This calls `ConverterAgent` to emit markdown and JSONL corpora in `data/processed/`.
