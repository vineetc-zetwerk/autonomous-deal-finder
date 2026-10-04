"""20% specialist + 80% RAG price ensemble, with course-evaluator support.

From this project's root:
    uv run python ensemble.py "Item details"
    uv run python ensemble.py --evaluate-first-200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import modal
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_url

from rag.local_store import open_local_collection, query_neighbors
from rag.shared_endpoint import price_from_neighbors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "ensemble_outputs"
RUN_VERSION = "single_user_v1_s20_r80"
DATASET_ID = "chitlangiavineet09/llama-3-2-price-prediction-data"
TEST_FILE = "data/fine_tune_test.jsonl"
QUESTION = "What does this cost to the nearest dollar?"
PREFIX = QUESTION + "\n\n"
SPECIALIST_WEIGHT = Decimal("0.20")
RAG_WEIGHT = Decimal("0.80")


@dataclass(frozen=True)
class TestItem:
    """The small predictor input interface expected by evaluation.py."""

    row_index: int
    title: str
    description: str
    price: float


def whole_usd(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def combine_prices(specialist_price: int, rag_price: int) -> int:
    if specialist_price < 0 or rag_price < 0:
        raise ValueError("Model prices must be nonnegative.")
    weighted = SPECIALIST_WEIGHT * Decimal(specialist_price) + RAG_WEIGHT * Decimal(rag_price)
    return whole_usd(weighted)


def parse_test_row(row: dict, position: int) -> TestItem:
    if row.get("row_index") != position:
        raise ValueError(f"Test row {position} has an unexpected row_index.")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"Test row {position} must contain one user message.")
    prompt = messages[0].get("content", "")
    if not isinstance(prompt, str) or not prompt.startswith(PREFIX):
        raise ValueError(f"Test row {position} has an unexpected prompt prefix.")
    description = prompt[len(PREFIX):].strip()
    if not re.match(r"^Title:\s*\S", description):
        raise ValueError(f"Test row {position} has no item title.")
    try:
        price = Decimal(str(row["expected_price"]))
    except (KeyError, InvalidOperation) as error:
        raise ValueError(f"Test row {position} has an invalid price label.") from error
    if not price.is_finite() or price <= 0:
        raise ValueError(f"Test row {position} needs a positive finite price.")
    title = description.splitlines()[0].removeprefix("Title:").strip()
    return TestItem(position, title, description, float(price))


def load_test_items(limit: int = 200) -> list[TestItem]:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required in the project .env to read the test file.")
    url = hf_hub_url(DATASET_ID, TEST_FILE, repo_type="dataset")
    items: list[TestItem] = []
    with requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=60
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            items.append(parse_test_row(json.loads(raw_line), len(items)))
            if len(items) == limit:
                break
    if len(items) != limit:
        raise RuntimeError(f"Expected {limit} test rows, found {len(items)}.")
    return items


class TwoModelEnsemble:
    def __init__(self, output_dir: Path | None = None):
        # Fail before making cloud calls if the required local index is absent.
        open_local_collection()
        self.preprocessor = modal.Cls.from_name("week8-price-pipeline", "Preprocessor")()
        self.specialist = modal.Cls.from_name("week8-price-pipeline", "Specialist")()
        self.rag = modal.Cls.from_name("week8-rag-price", "RagPredictor")()
        self.endpoint_url = os.getenv("MODAL_SHARED_ENDPOINT_URL", "").strip()
        self.endpoint_model = os.getenv("MODAL_SHARED_ENDPOINT_MODEL", "").strip()
        if not self.endpoint_url or not self.endpoint_model:
            raise RuntimeError("Set the Modal Shared Endpoint URL and model in the project .env.")
        self.output_dir = output_dir
        self.cache_path = output_dir / "predictions.jsonl" if output_dir else None
        self.lock = threading.Lock()
        self.cache: dict[int, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                record = json.loads(line)
                if record.get("run_version") != RUN_VERSION:
                    raise RuntimeError("Cached predictions used a different prompt or weighting.")
                self.cache[record["row_index"]] = record

    def predict_description(self, description: str, row_index: int | None = None) -> dict:
        digest = hashlib.sha256(description.encode()).hexdigest()
        if row_index is not None:
            with self.lock:
                cached = self.cache.get(row_index)
            if cached is not None:
                if cached["description_sha256"] != digest:
                    raise RuntimeError(f"Cached test row {row_index} does not match this dataset.")
                return cached

        # Both calls consume the same standardized description. Only vector
        # comparison and metadata retrieval run locally.
        with ThreadPoolExecutor(max_workers=2) as pool:
            specialist_future = pool.submit(self.specialist.price.remote, description)
            embedding_future = pool.submit(self.rag.embed.remote, description)
            specialist_price = int(specialist_future.result())
            embedding = embedding_future.result()
        neighbors = query_neighbors(embedding)
        rag_result = price_from_neighbors(
            description, neighbors, self.endpoint_url, self.endpoint_model
        )
        rag_price = int(rag_result["price_usd"])
        record = {
            "run_version": RUN_VERSION,
            "row_index": row_index,
            "description_sha256": digest,
            "specialist_price_usd": specialist_price,
            "rag_price_usd": rag_price,
            "ensemble_price_usd": combine_prices(specialist_price, rag_price),
            "neighbor_ids": [item["id"] for item in neighbors],
        }
        if row_index is not None and self.cache_path is not None:
            with self.lock:
                # A completed line is recoverable if a later cloud call fails.
                with self.cache_path.open("a") as output:
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                self.cache[row_index] = record
        return record

    def predict_test_item(self, item: TestItem) -> int:
        # Test rows are already in the exact training prompt format. Never pass
        # item.price to either model; the evaluator alone uses that label.
        return self.predict_description(item.description, item.row_index)["ensemble_price_usd"]

    def predict_raw(self, text: str) -> dict:
        description = self.preprocessor.normalize.remote(text)
        return {"description": description, **self.predict_description(description)}


def run_evaluation(output_dir: Path, workers: int) -> None:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from plotly.graph_objects import Figure
    from evaluation import evaluate

    items = load_test_items(200)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = TwoModelEnsemble(output_dir)
    chart_paths = [output_dir / "error_trend.html", output_dir / "actual_vs_predicted.html"]
    original_show = Figure.show
    next_chart = 0

    def save_chart(figure: Figure, *args, **kwargs) -> None:
        nonlocal next_chart
        path = chart_paths[next_chart]
        figure.write_html(path, include_plotlyjs=True)
        print(f"\nSaved evaluator chart: {path}", flush=True)
        next_chart += 1

    try:
        Figure.show = save_chart
        evaluate(ensemble.predict_test_item, items, size=200, workers=workers)
    finally:
        Figure.show = original_show

    records = [ensemble.cache[item.row_index] for item in items]
    truths = [item.price for item in items]
    guesses = [record["ensemble_price_usd"] for record in records]
    results_path = output_dir / "results.csv"
    with results_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "row_index", "title", "actual_price_usd", "specialist_price_usd",
            "rag_price_usd", "ensemble_price_usd", "absolute_error_usd",
        ])
        writer.writeheader()
        for item, record in zip(items, records):
            writer.writerow({
                "row_index": item.row_index,
                "title": item.title,
                "actual_price_usd": item.price,
                "specialist_price_usd": record["specialist_price_usd"],
                "rag_price_usd": record["rag_price_usd"],
                "ensemble_price_usd": record["ensemble_price_usd"],
                "absolute_error_usd": abs(record["ensemble_price_usd"] - item.price),
            })
    summary = {
        "count": len(items),
        "weights": {"specialist": float(SPECIALIST_WEIGHT), "rag": float(RAG_WEIGHT)},
        "mae_usd": mean_absolute_error(truths, guesses),
        "mse_usd_squared": mean_squared_error(truths, guesses),
        "rmse_usd": mean_squared_error(truths, guesses) ** 0.5,
        "r2": r2_score(truths, guesses),
        "results_csv": str(results_path),
        "charts": [str(path) for path in chart_paths],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Raw item details to price")
    parser.add_argument("--evaluate-first-200", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.evaluate_first_200 == bool(args.text):
        parser.error("Provide either item text or --evaluate-first-200.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    load_dotenv(HERE / ".env", override=False)
    if HERE.name == "myfiles" and HERE.parent.name == "week8":
        load_dotenv(HERE.parent / ".env", override=False)
    if args.evaluate_first_200:
        run_evaluation(OUTPUT_DIR / RUN_VERSION, args.workers)
    else:
        result = TwoModelEnsemble().predict_raw(args.text)
        print("Canonical item description:\n" + result["description"])
        print(f"Specialist: ${result['specialist_price_usd']}")
        print(f"RAG: ${result['rag_price_usd']}")
        print(f"Ensemble (20% specialist / 80% RAG): ${result['ensemble_price_usd']}")


if __name__ == "__main__":
    main()
