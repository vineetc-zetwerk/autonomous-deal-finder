"""Read the downloaded Chroma index and search it with a Modal-made vector."""

from __future__ import annotations

import json
import math
from pathlib import Path

import chromadb


INDEX_DIR = Path(__file__).resolve().parent / "vector_db" / "minilm_l6_v2_train_v1"
COLLECTION = "training_items"
EXPECTED_ROWS = 20_000
EXPECTED_DIMENSIONS = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def open_local_collection(index_dir: Path = INDEX_DIR):
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file() or not (index_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"Local Chroma index is missing at {index_dir}. "
            "Download it using the command in README.md."
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("rows") != EXPECTED_ROWS
        or manifest.get("collection") != COLLECTION
        or manifest.get("embedding_model") != EMBEDDING_MODEL
    ):
        raise RuntimeError("Local Chroma manifest does not match the RAG predictor.")
    collection = chromadb.PersistentClient(path=str(index_dir)).get_collection(
        COLLECTION, embedding_function=None
    )
    if collection.count() != EXPECTED_ROWS:
        raise RuntimeError("Local Chroma index has an incomplete row count.")
    return collection


def query_neighbors(embedding: list[float], index_dir: Path = INDEX_DIR) -> list[dict]:
    """Run vector comparison locally; no embedding model is loaded here."""
    if len(embedding) != EXPECTED_DIMENSIONS or not all(math.isfinite(value) for value in embedding):
        raise ValueError(f"Expected {EXPECTED_DIMENSIONS} finite embedding values.")
    result = open_local_collection(index_dir).query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    neighbors = [
        {
            "id": identifier,
            "description": document,
            "price_usd": float(metadata["price_usd"]),
            "distance": float(distance),
        }
        for identifier, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0]
        )
    ]
    if len(neighbors) != 5:
        raise RuntimeError("Local Chroma returned fewer than five neighbors.")
    return neighbors
