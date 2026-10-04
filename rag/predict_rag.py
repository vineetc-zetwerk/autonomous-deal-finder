"""Embed on Modal, search local Chroma, then price with a Modal endpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import modal
from dotenv import load_dotenv

from local_store import query_neighbors
from shared_endpoint import price_from_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG price estimate from five training items.")
    parser.add_argument("text", help="Raw item title and details")
    parser.add_argument(
        "--neighbors-only", action="store_true",
        help="Show Chroma matches without calling the Qwen Shared Endpoint",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    if project_root.name == "myfiles" and project_root.parent.name == "week8":
        load_dotenv(project_root.parent / ".env", override=False)

    preprocessor = modal.Cls.from_name("week8-price-pipeline", "Preprocessor")()
    rag = modal.Cls.from_name("week8-rag-price", "RagPredictor")()
    description = preprocessor.normalize.remote(args.text)
    print("Canonical item description:\n" + description)
    embedding = rag.embed.remote(description)
    neighbors = query_neighbors(embedding)

    if args.neighbors_only:
        for number, item in enumerate(neighbors, start=1):
            print(
                f"\nReference {number}: {item['id']} "
                f"(price ${item['price_usd']:.2f}, cosine distance {item['distance']:.4f})\n"
                + item["description"]
            )
        return

    endpoint_model = os.getenv("MODAL_SHARED_ENDPOINT_MODEL", "").strip()
    endpoint_url = os.getenv("MODAL_SHARED_ENDPOINT_URL", "").strip()
    if not endpoint_model or not endpoint_url:
        raise SystemExit(
            "Set MODAL_SHARED_ENDPOINT_MODEL and MODAL_SHARED_ENDPOINT_URL "
            "in the project .env from your Modal Shared Endpoint dashboard."
        )
    result = price_from_neighbors(description, neighbors, endpoint_url, endpoint_model)
    print(f"RAG predicted price: ${result['price_usd']}")
    print("References used:")
    for item in result["neighbors"]:
        print(f"  {item['id']}: ${item['price_usd']:.2f} — {item['description'].splitlines()[0]}")


if __name__ == "__main__":
    main()
