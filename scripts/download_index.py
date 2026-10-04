"""Download the project's complete Chroma index from its Modal Volume."""

from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_VOLUME_NAME = "week8-rag-chroma"
INDEX_VOLUME_PATH = "/minilm_l6_v2_train_v1"


def download_index(project_root: Path = PROJECT_ROOT) -> Path:
    destination = project_root / "rag" / "vector_db"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", INDEX_VOLUME_NAME, INDEX_VOLUME_PATH, str(destination)],
        check=True,
    )
    return destination / INDEX_VOLUME_PATH.lstrip("/")


if __name__ == "__main__":
    print(f"Downloaded Chroma index to {download_index()}")
