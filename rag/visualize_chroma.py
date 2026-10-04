"""Plot the local Chroma index and export the first five full vectors.

Run from this project's root:
    uv run python rag/visualize_chroma.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Keep plot/font caches in a writable temporary directory on local machines.
cache_dir = Path(tempfile.gettempdir()) / "week8-rag-plot-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

from local_store import EXPECTED_DIMENSIONS, EXPECTED_ROWS, open_local_collection

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIRST_IDS = [f"train-{number:05d}" for number in range(5)]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    collection = open_local_collection()
    vectors = np.empty((EXPECTED_ROWS, EXPECTED_DIMENSIONS), dtype=np.float32)
    prices = np.empty(EXPECTED_ROWS, dtype=np.float32)
    positions: dict[str, int] = {}

    for offset in range(0, EXPECTED_ROWS, 1000):
        batch = collection.get(
            limit=min(1000, EXPECTED_ROWS - offset),
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        ids = batch["ids"]
        embeddings = np.asarray(batch["embeddings"], dtype=np.float32)
        if embeddings.shape != (len(ids), EXPECTED_DIMENSIONS):
            raise RuntimeError(f"Unexpected embedding shape at offset {offset}: {embeddings.shape}")
        end = offset + len(ids)
        vectors[offset:end] = embeddings
        prices[offset:end] = [float(meta["price_usd"]) for meta in batch["metadatas"]]
        positions.update((item_id, offset + index) for index, item_id in enumerate(ids))

    if len(positions) != EXPECTED_ROWS or not all(item_id in positions for item_id in FIRST_IDS):
        raise RuntimeError("The local index has missing or duplicate item IDs.")

    first = collection.get(ids=FIRST_IDS, include=["documents", "metadatas"])
    first_by_id = {
        item_id: {
            "id": item_id,
            "description": first["documents"][index],
            "price_usd": float(first["metadatas"][index]["price_usd"]),
            "embedding_dimensions": EXPECTED_DIMENSIONS,
            "embedding": vectors[positions[item_id]].astype(float).tolist(),
        }
        for index, item_id in enumerate(first["ids"])
    }
    first_five = [first_by_id[item_id] for item_id in FIRST_IDS]

    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    points = pca.fit_transform(vectors)
    fig, axis = plt.subplots(figsize=(12, 8), layout="constrained")
    scatter = axis.scatter(
        points[:, 0], points[:, 1], c=np.log1p(prices), cmap="viridis",
        s=6, alpha=0.6, linewidths=0, rasterized=True,
    )
    for number, item_id in enumerate(FIRST_IDS):
        x, y = points[positions[item_id]]
        axis.scatter(x, y, marker="o", s=100, facecolors="none", edgecolors="red", linewidths=2)
        axis.annotate(str(number), (x, y), xytext=(7, 7), textcoords="offset points", color="red", weight="bold")
    axis.set_title(f"Chroma training-item embeddings ({EXPECTED_ROWS:,} items; PCA projection)")
    axis.set_xlabel(f"Principal component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axis.set_ylabel(f"Principal component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis).set_label("log(1 + training price in USD)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "chroma_embeddings_pca.png"
    vectors_path = OUTPUT_DIR / "first_five_embeddings.json"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    vectors_path.write_text(json.dumps(first_five, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved scatter plot: {plot_path}")
    print(f"Saved full first-five vectors: {vectors_path}")


if __name__ == "__main__":
    main()
