"""
Vector Store Agent
------------------
Loads converted text documents into a ChromaDB vector database
for semantic retrieval by the LLM query pipeline.

Workflow:
1. Read text files from data/processed/text/
2. Split each document into overlapping chunks
3. Generate embeddings for each chunk
4. Store in ChromaDB with metadata
5. Support semantic search queries

The resulting database allows the LLM to retrieve only the most
relevant chunks when answering questions about chanterelle habitat,
soil conditions, hydrology, vegetation etc.
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

from config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHUNKS_PER_DOC,
    FETCH_OUTPUT_DIR
)

TEXT_DIR = "data/processed/text"
FETCH_DIR = "data/raw/fetched"


class VectorStoreAgent:

    def __init__(self):
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)

        # Persistent ChromaDB client - data survives restarts
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR
        )

        # Use sentence transformers for embeddings
        # Same model as classifier for consistency
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' "
            f"loaded. Current chunks: "
            f"{self.collection.count()}"
        )

    # ----------------------------------------------------------
    # MAIN INGEST METHOD
    # ----------------------------------------------------------

    def ingest_all(self, force_reingest=False):
        """
        Load all converted text documents into ChromaDB.
        Skips documents already ingested unless force_reingest=True.
        """
        text_files = sorted([
            f for f in os.listdir(TEXT_DIR)
            if f.endswith(".txt")
        ])

        print(f"\nFound {len(text_files)} text documents to ingest")
        print(f"Chunk size: {CHUNK_SIZE} chars, "
              f"overlap: {CHUNK_OVERLAP} chars\n")

        ingested = 0
        skipped = 0
        failed = 0
        total_chunks = 0

        for i, filename in enumerate(text_files):
            # Dataset ID from filename
            dataset_id = filename.replace(".txt", "")
            filepath = os.path.join(TEXT_DIR, filename)

            # Check if already ingested
            if not force_reingest and self._already_ingested(
                dataset_id
            ):
                skipped += 1
                continue

            print(f"[{i+1}/{len(text_files)}] {dataset_id[:60]}")

            try:
                # Load text
                with open(filepath, encoding="utf-8") as f:
                    text = f.read()

                # Load sidecar metadata if available
                meta = self._load_sidecar(dataset_id)

                # Chunk the document
                chunks = self._chunk_text(text, dataset_id)

                if not chunks:
                    print(f"  No chunks generated, skipping")
                    failed += 1
                    continue

                # Apply safety cap
                if len(chunks) > MAX_CHUNKS_PER_DOC:
                    print(
                        f"  Capping at {MAX_CHUNKS_PER_DOC} chunks "
                        f"(was {len(chunks)})"
                    )
                    chunks = chunks[:MAX_CHUNKS_PER_DOC]

                # Add to ChromaDB
                self._add_chunks(chunks, dataset_id, meta)

                print(
                    f"  ✓ {len(chunks)} chunks ingested"
                )
                total_chunks += len(chunks)
                ingested += 1

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed += 1

        print()
        print("=" * 60)
        print("VECTOR STORE INGEST SUMMARY")
        print("=" * 60)
        print(f"  Documents ingested:  {ingested}")
        print(f"  Documents skipped:   {skipped}")
        print(f"  Documents failed:    {failed}")
        print(f"  New chunks added:    {total_chunks}")
        print(f"  Total chunks in DB:  {self.collection.count()}")

    # ----------------------------------------------------------
    # CHUNKING
    # ----------------------------------------------------------

    def _chunk_text(self, text, dataset_id):
        """
        Split text into overlapping chunks.
        Tries to split on section boundaries first (===)
        then falls back to character-level chunking.
        """
        chunks = []

        # First try to split on section headers
        # This keeps dataset record, schema, spatial summary
        # and feature descriptions as coherent units
        sections = text.split("===")

        if len(sections) > 2:
            # Reassemble section headers with their content
            section_texts = []
            for j in range(1, len(sections), 2):
                if j < len(sections):
                    header = sections[j].strip()
                    content = sections[j+1].strip() if j+1 < len(sections) else ""
                    section_texts.append(f"=== {header} ===\n{content}")

            # Now chunk within each section
            for section_text in section_texts:
                section_chunks = self._character_chunk(section_text)
                chunks.extend(section_chunks)
        else:
            # No section structure, chunk the whole thing
            chunks = self._character_chunk(text)

        return chunks

    def _character_chunk(self, text):
        """Split text into overlapping character-level chunks."""
        chunks = []

        if len(text) <= CHUNK_SIZE:
            if text.strip():
                chunks.append(text.strip())
            return chunks

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE

            # Try to end on a newline for cleaner chunks
            if end < len(text):
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + (CHUNK_SIZE // 2):
                    end = newline_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward by chunk size minus overlap
            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break

        return chunks

    # ----------------------------------------------------------
    # CHROMADB OPERATIONS
    # ----------------------------------------------------------

    def _add_chunks(self, chunks, dataset_id, meta):
        """Add a list of text chunks to ChromaDB."""

        # Remove existing chunks for this dataset
        # so reingest works cleanly
        try:
            existing = self.collection.get(
                where={"dataset_id": dataset_id}
            )
            if existing["ids"]:
                self.collection.delete(
                    where={"dataset_id": dataset_id}
                )
        except Exception:
            pass

        # Build IDs, documents, and metadata for each chunk
        ids = []
        documents = []
        metadatas = []

        for j, chunk in enumerate(chunks):
            chunk_id = f"{dataset_id}_chunk_{j}"

            # Metadata for filtering and attribution
            chunk_meta = {
                "dataset_id": dataset_id,
                "chunk_index": j,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now().isoformat()[:10]
            }

            # Add sidecar metadata if available
            if meta:
                chunk_meta["title"] = meta.get("title", dataset_id)
                chunk_meta["source_type"] = meta.get(
                    "source_type", "unknown"
                )
                chunk_meta["feature_count"] = meta.get(
                    "feature_count", 0
                )

                # Add geometry type as string for filtering
                geom_types = meta.get("geometry_types", [])
                if geom_types:
                    chunk_meta["geometry_type"] = geom_types[0]

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_meta)

        # ChromaDB has a batch size limit
        # Add in batches of 100 to be safe
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end]
            )

    def _already_ingested(self, dataset_id):
        """Check if this dataset already has chunks in the DB."""
        try:
            results = self.collection.get(
                where={"dataset_id": dataset_id},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    def _load_sidecar(self, dataset_id):
        """Load the metadata sidecar for a dataset if it exists."""
        # Reconstruct the sidecar filename
        safe_id = dataset_id.replace(":", "_")
        sidecar_path = os.path.join(
            FETCH_DIR, f"{safe_id}_meta.json"
        )

        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                return json.load(f)
        return None

    # ----------------------------------------------------------
    # SEMANTIC SEARCH
    # ----------------------------------------------------------

    def search(self, query, n_results=8, filter_meta=None):
        """
        Search the vector database for chunks relevant
        to a natural language query.

        Returns a list of result dicts with text and metadata.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self.collection.count())
        }

        if filter_meta:
            kwargs["where"] = filter_meta

        results = self.collection.query(**kwargs)

        formatted = []
        for j in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][j],
                "text": results["documents"][0][j],
                "metadata": results["metadatas"][0][j],
                "distance": results["distances"][0][j]
            })

        return formatted

    def search_and_print(self, query, n_results=5):
        """Search and print results in readable format."""
        print(f"\nQuery: '{query}'")
        print("=" * 60)

        results = self.search(query, n_results=n_results)

        for i, r in enumerate(results):
            meta = r["metadata"]
            title = meta.get("title", meta.get("dataset_id", "?"))
            distance = r["distance"]
            relevance = 1 - distance

            print(f"\n[{i+1}] {title}")
            print(f"     Relevance: {relevance:.3f}")
            print(f"     Chunk: {meta.get('chunk_index', '?')}/"
                  f"{meta.get('total_chunks', '?')}")
            print(f"     Text preview:")
            print(f"     {r['text'][:300]}...")

    def get_stats(self):
        """Print database statistics."""
        count = self.collection.count()
        print(f"\nChromaDB Statistics:")
        print(f"  Collection: {CHROMA_COLLECTION_NAME}")
        print(f"  Total chunks: {count}")
        print(f"  Storage: {CHROMA_DB_DIR}")


if __name__ == "__main__":

    agent = VectorStoreAgent()
    agent.ingest_all()
    agent.get_stats()

    # Test searches relevant to chanterelle habitat
    test_queries = [
        "soil moisture and drainage conditions near Ucluelet",
        "coniferous forest species Douglas-fir hemlock cedar",
        "biogeoclimatic zone coastal western hemlock",
        "terrain slope aspect elevation",
        "chanterelle mushroom fungal habitat",
        "precipitation rainfall coastal BC",
        "old growth forest ancient trees",
        "aquifer groundwater hydrology watershed"
    ]

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH TESTS")
    print("=" * 60)

    for query in test_queries:
        agent.search_and_print(query, n_results=3)
        print()