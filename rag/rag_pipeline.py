"""Build the Chroma index and run query embedding/price inference on Modal.

    uv run modal run rag/rag_pipeline.py::build_index
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlparse

import modal


DATASET_ID = "chitlangiavineet09/llama-3-2-price-prediction-data"
DATASET_FILE = "data/fine_tune_train.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "training_items"
EXPECTED_ROWS = 20_000
INDEX_MOUNT = "/index"
INDEX_NAME = "minilm_l6_v2_train_v1"
QUESTION = "What does this cost to the nearest dollar?"
FIELDS = ("Title", "Category", "Brand", "Description", "Details")

if modal.is_local():
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    if project_root.name == "myfiles" and project_root.parent.name == "week8":
        load_dotenv(project_root.parent / ".env", override=False)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    proxy_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
    proxy_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required in the project .env to read the dataset.")
    hf_auth = modal.Secret.from_dict({"HF_TOKEN": hf_token})
    proxy_auth = modal.Secret.from_dict(
        {"MODAL_PROXY_TOKEN_ID": proxy_id, "MODAL_PROXY_TOKEN_SECRET": proxy_secret}
    )
else:
    hf_auth = modal.Secret.from_dict({})
    proxy_auth = modal.Secret.from_dict({})

app = modal.App("week8-rag-price")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.5.1",
    "transformers==4.56.2",
    "sentence-transformers==5.1.1",
    "huggingface_hub==0.34.4",
    "chromadb==1.3.4",
    "requests==2.34.2",
)
index_volume = modal.Volume.from_name("week8-rag-chroma", create_if_missing=True)
model_cache = modal.Volume.from_name("week8-rag-model-cache", create_if_missing=True)
image = image.env({"HF_HOME": "/model-cache"})


def description_from_training_row(row: dict) -> tuple[str, float]:
    """Return only Title..Details for embedding, and its labeled price."""
    messages = row.get("messages", [])
    if len(messages) != 2 or [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError("Training row needs user and assistant messages.")
    prompt = messages[0].get("content", "")
    prefix = QUESTION + "\n\n"
    if not isinstance(prompt, str) or not prompt.startswith(prefix):
        raise ValueError("Training row has an unexpected user prompt.")
    description = prompt[len(prefix):].strip()
    if not description.startswith("Title:"):
        raise ValueError("Training item does not start with Title.")
    try:
        price = Decimal(str(messages[1]["content"]))
    except (KeyError, InvalidOperation) as error:
        raise ValueError("Training row has an invalid price label.") from error
    if not price.is_finite() or price < 0:
        raise ValueError("Training row price must be nonnegative and finite.")
    return description, float(price)


def validate_description(description: str) -> None:
    """Check the canonical field boundary without rewriting training data."""
    if not isinstance(description, str):
        raise ValueError("Description must be text.")
    # Some training items have an empty Brand field; preserve their source text.
    found = re.findall(r"(?m)^(Title|Category|Brand|Description|Details):", description)
    if tuple(found) != FIELDS:
        raise ValueError("Description must contain Title through Details in order.")


def parse_rag_price(answer: str) -> int:
    """Extract a whole-USD amount from the shared model's final response."""
    final = re.sub(r"(?s)<think>.*?</think>\s*", "", answer).strip()
    # Some endpoint responses wrap an otherwise valid single price in Markdown.
    final = final.strip("*` ").strip()
    match = re.fullmatch(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", final)
    if not match:
        raise ValueError(f"RAG model returned an invalid price: {final!r}")
    amount = Decimal(match.group(1).replace(",", ""))
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def rag_messages(description: str, neighbors: list[dict]) -> list[dict[str, str]]:
    validate_description(description)
    if len(neighbors) != 5:
        raise ValueError("RAG prediction requires exactly five neighbors.")
    examples = "\n\n".join(
        f"Reference {number} (known price ${item['price_usd']:.2f}):\n{item['description']}"
        for number, item in enumerate(neighbors, start=1)
    )
    return [
        {
            "role": "user",
            "content": (
                "Estimate the price of this product. Respond with the price, no explanation\n\n"
                f"{description}\n\n"
                "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
                f"{examples}"
            ),
        },
    ]


@app.function(
    image=image,
    gpu="T4",
    secrets=[hf_auth],
    volumes={INDEX_MOUNT: index_volume, "/model-cache": model_cache},
    timeout=3600,
)
def create_index() -> dict:
    """Embed all 20,000 training rows once; never index validation/test rows."""
    import chromadb
    import requests
    from huggingface_hub import hf_hub_url
    from sentence_transformers import SentenceTransformer

    target = Path(INDEX_MOUNT) / INDEX_NAME
    manifest_path = target / "manifest.json"
    if target.exists():
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("rows") == EXPECTED_ROWS and manifest.get("embedding_model") == EMBEDDING_MODEL:
                return manifest
        raise RuntimeError(f"Existing index is incomplete: {target}")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    url = hf_hub_url(DATASET_ID, DATASET_FILE, repo_type="dataset")
    with tempfile.TemporaryDirectory() as temporary:
        local_index = Path(temporary) / "chroma"
        client = chromadb.PersistentClient(path=str(local_index))
        collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        docs: list[str] = []
        prices: list[float] = []
        ids: list[str] = []
        count = 0

        def flush() -> None:
            if not docs:
                return
            embeddings = model.encode(
                docs, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            )
            collection.add(
                ids=ids,
                documents=docs,
                embeddings=embeddings.tolist(),
                metadatas=[{"price_usd": price, "split": "train"} for price in prices],
            )
            docs.clear()
            prices.clear()
            ids.clear()

        with requests.get(
            url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]},
            stream=True, timeout=60,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                description, price = description_from_training_row(json.loads(raw_line))
                docs.append(description)
                prices.append(price)
                ids.append(f"train-{count:05d}")
                count += 1
                if len(docs) == 256:
                    flush()
                    if count % 2560 == 0:
                        print(f"Embedded {count:,} of {EXPECTED_ROWS:,} training items", flush=True)
        flush()
        if count != EXPECTED_ROWS or collection.count() != EXPECTED_ROWS:
            raise RuntimeError(f"Index has {collection.count()} of {EXPECTED_ROWS} expected rows.")

        # Chroma's SQLite files are copied only after indexing is complete.
        # Inference containers receive a read-only snapshot in local scratch.
        shutil.copytree(local_index, target)
        manifest = {
            "rows": count,
            "dataset": DATASET_ID,
            "source_file": DATASET_FILE,
            "embedding_model": EMBEDDING_MODEL,
            "collection": COLLECTION,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        index_volume.commit()
        return manifest


@app.cls(
    image=image,
    gpu="T4",
    secrets=[proxy_auth],
    volumes={"/model-cache": model_cache},
    timeout=600,
)
class RagPredictor:
    @modal.enter()
    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    @modal.method()
    def embed(self, description: str) -> list[float]:
        """Embed one canonical item; its vector is searched locally by the caller."""
        validate_description(description)
        embedding = self.model.encode([description], normalize_embeddings=True)[0]
        return embedding.tolist()

    @modal.method()
    def price(
        self, description: str, neighbors: list[dict], endpoint_url: str, endpoint_model: str
    ) -> dict:
        import requests

        if not endpoint_model.strip():
            raise ValueError("A Modal Shared Endpoint model ID is required.")
        parsed_url = urlparse(endpoint_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or not parsed_url.hostname.endswith(".modal.direct")
            or parsed_url.path != "/v1/chat/completions"
        ):
            raise ValueError("Expected a Modal HTTPS chat-completions endpoint URL.")
        proxy_id = os.environ.get("MODAL_PROXY_TOKEN_ID", "")
        proxy_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "")
        if not proxy_id or not proxy_secret:
            raise RuntimeError("Modal proxy token is missing from the deployed app.")
        if len(neighbors) != 5:
            raise ValueError("Expected five local Chroma neighbors.")
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {proxy_id}.{proxy_secret}"},
            json={
                "model": endpoint_model,
                "messages": rag_messages(description, neighbors),
                "max_tokens": 1024,
                "reasoning_effort": "low",
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"] or ""
        return {"price_usd": parse_rag_price(answer), "neighbors": neighbors}


@app.local_entrypoint()
def build_index() -> None:
    print(json.dumps(create_index.remote(), indent=2))
