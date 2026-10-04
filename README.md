# Autonomous Deal Finder

## Use case

Online deal feeds contain many products, but a listed discount does not tell you
whether a price is genuinely attractive. This project is an on-demand deal
finder: one click scans current electronics, computer, and smart-home offers,
estimates the value of five well-described products, and highlights the deal
with the largest **positive percentage saving** against its estimated price.
It shows the reasoning steps and final announcement in a Gradio UI. It does
not purchase products or send email, push notifications, or messages.

## Tools and components

| Component | Role |
| --- | --- |
| Planning Agent | Calls the scanner, ensemble, and messaging tools in sequence through Qwen tool calling. Python verifies that all five deals were estimated and selects the winner. |
| Scanner Agent | Parses the first 10 entries from each of three DealNews RSS feeds (30 total); Qwen selects five detailed products with clear prices. Pydantic and source-price checks validate the selection. |
| Price Ensemble | Produces a whole-USD estimate from **20% specialist + 80% RAG** predictions. |
| Messaging Agent | Uses Qwen for a short product sentence; Python attaches the exact prices, savings, and deal URL. The announcement stays in the UI. |
| Gradio | Left panel: status, comparisons, and winning deal. Right panel: timestamped agent requests and responses, without authorization headers or tokens. |

### Flow

```text
Click "Find the best deal now"
    → Planning Agent calls Scanner Agent
    → 3 RSS feeds × 10 items → 30 parsed deals → 5 validated deals
    → for each of the 5 deals, Planning Agent calls Price Ensemble
         raw description → Llama preprocessor → canonical five-field description
         ├─ fine-tuned Llama specialist → price A
         └─ MiniLM query embedding → local Chroma top 5 → Qwen RAG → price B
         expected price = round(0.20 × A + 0.80 × B)
    → Python ranks positive (expected − deal) / expected percentages
    → Planning Agent calls Messaging Agent for the single winner
    → announcement, deal link, comparisons, and live log appear in Gradio
```

The scanner chooses candidates for description quality and a defensible price,
not for the size of a claimed discount. The planner compares only those five
candidates. If none is below its estimated price, it displays no winning deal.
The three sources are DealNews [Electronics](https://www.dealnews.com/c142/Electronics/?rss=1),
[Computers](https://www.dealnews.com/c39/Computers/?rss=1), and
[Smart Home](https://www.dealnews.com/f1912/Smart-Home/?rss=1) RSS feeds.

## Models and data

| Purpose | Model or data | Where it runs |
| --- | --- | --- |
| Standardize raw product details | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Modal GPU |
| Specialist price predictor | Same Llama base + [price-predictor QLoRA adapter](https://huggingface.co/chitlangiavineet09/llama-3.2-3b-price-predictor-qlora) | Modal GPU |
| Build and query embeddings | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), 384 dimensions | Modal GPU |
| Similarity search | Chroma, 20,000 training-item descriptions and price metadata | Local disk; no model inference during vector comparison |
| RAG pricing, scanning, planning, announcement wording | `Qwen/Qwen3.8-2.4T-A95B` via a configured Modal Shared Endpoint | Modal |
| Training/evaluation source | [Hugging Face price-prediction dataset](https://huggingface.co/datasets/chitlangiavineet09/llama-3-2-price-prediction-data) | Training split indexed; test split used only for evaluation |

The shared description contract is:

```text
Title: ...
Category: ...
Brand: ...
Description: ...
Details: ...
```

The known training prices live in Chroma metadata, not in the embedded text.
Each query is embedded on Modal, compared locally with the Chroma index, and
sent with the five nearest items and their known prices to Qwen for a RAG
estimate. The test item's known price is never passed to either predictor.

## Project layout

```text
.
├── autonomous/       # Qwen tool-calling planner and two-panel Gradio app
├── scanner/          # RSS collection, deal validation, structured selection
├── specialist/       # Modal Llama preprocessor and fine-tuned price predictor
├── rag/              # Modal MiniLM, local Chroma search, Qwen RAG, visualization
├── messaging/        # Deal announcement writer
├── scripts/          # Download the Chroma index from the Modal Volume
├── ensemble.py       # 20/80 price blend and test-set runner
├── evaluation.py     # Self-contained metrics and charts
├── pyproject.toml    # Standalone Python dependencies
├── uv.lock           # Reproducible local dependency versions
└── .env.example      # Credential names only; copy to an ignored .env
```

Generated scan snapshots, evaluation runs, visualization files, and the Chroma
index stay in their component folders locally but are excluded from Git. The
index must be built or downloaded by each user; it is not committed.

## Setup

After cloning the repository, run these commands from its root. Use Python
3.12 and [uv](https://docs.astral.sh/uv/). You also need your own Modal
account, access to the gated Llama base model on Hugging Face, and a
[Modal Shared Endpoint](https://modal.com/docs/guide/shared-endpoints) for
the configured Qwen model. The endpoint, Modal GPU jobs, and model downloads
may incur charges.

```bash
uv sync --locked
cp .env.example .env
```

Edit `.env` with your own `HF_TOKEN`, Modal credentials, Modal proxy token,
endpoint model ID, and endpoint URL. Alternatively, use an authenticated local
Modal profile for account access. Never commit `.env`. The Llama adapter
requires the `Llama-3.2-3B-Instruct` base specified in the example.

### Modal configuration

The Modal app definitions **are included in this repository**:
`specialist/price_pipeline.py` defines the GPU preprocessor and fine-tuned
specialist; `rag/rag_pipeline.py` defines the GPU embedding service, index
builder, image dependencies, and Volumes. The other agents call your Qwen
Shared Endpoint through the URL and model ID in `.env`.

Your Modal account credentials, proxy token, live Shared Endpoint, deployed
apps, and Volume contents **are not included**. Create a Shared Endpoint in
your own Modal dashboard, copy its URL and model ID into `.env`, and put its
proxy token ID and secret in the corresponding fields. Use a local Modal
profile or uncomment and fill `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in
`.env`. Verify account access before deploying:

```bash
uv run --env-file .env modal token info
```

This repository contains all application source and setup commands, but a
clone cannot perform cloud inference until its owner supplies credentials,
creates the Shared Endpoint, and deploys the included Modal apps.

Deploy the two Modal apps:

```bash
uv run --env-file .env modal deploy specialist/price_pipeline.py
uv run --env-file .env modal deploy rag/rag_pipeline.py
```

Create the Chroma index from the 20,000 training rows, then copy it to the
local path expected by the app:

```bash
uv run --env-file .env modal run rag/rag_pipeline.py::build_index
uv run --env-file .env python scripts/download_index.py
```

Index creation is idempotent if a complete index already exists on the Modal
Volume. The local copy is approximately 90 MB. To inspect it, run
`uv run --env-file .env python rag/visualize_chroma.py`; this writes a 2D PCA plot and the
first five full embedding vectors under the ignored `rag/outputs/` directory.

Start the UI:

```bash
uv run --env-file .env python autonomous/app.py
```

Open <http://127.0.0.1:7860>, then click **Find the best deal now**. Each
click scans the live RSS feeds again. Set `GRADIO_SERVER_PORT` if port 7860 is
occupied.

## Individual components and evaluation

```bash
# Scan 30 RSS entries without an LLM, or select five with Qwen:
uv run --env-file .env python scanner/scanner_agent.py --rss-only
uv run --env-file .env python scanner/scanner_agent.py

# Price one raw item through each path:
uv run --env-file .env python specialist/predict.py "HyperX QuadCast USB microphone"
uv run --env-file .env python rag/predict_rag.py "HyperX QuadCast USB microphone"
uv run --env-file .env python ensemble.py "HyperX QuadCast USB microphone"

# Evaluate the 20/80 ensemble on the first 200 test rows:
uv run --env-file .env python ensemble.py --evaluate-first-200 --workers 5
```

The evaluation runner downloads the test JSONL, keeps its price labels out of
the model input, and saves a CSV, MAE/MSE/RMSE/R² summary, and two HTML charts
under the ignored `ensemble_outputs/single_user_v1_s20_r80/` directory. It is
resumable by test-row index but makes many paid inference calls on a first run.
The included `evaluation.py` computes metrics and saves the charts; no
evaluator from another repository is required.

Run the local test suites:

```bash
for folder in . autonomous scanner messaging specialist rag; do
  uv run --env-file .env python -m unittest discover -s "$folder" -p 'test_*.py'
done
```

The RAG suite includes a local Chroma integration test and therefore needs
the downloaded index.

## Boundaries and safety

- Estimates are model outputs, not appraisals or guaranteed savings. Confirm the
  current offer, product variant, shipping, tax, condition, and availability on
  the deal page before acting.
- RSS collection is deterministic; LLM inference starts when the scanner
  selects five deals. Product price evidence is checked against the RSS text,
  and amounts described as “$X off” are not treated as the sale price.
- Only the vector database lookup runs locally during RAG prediction. Llama,
  MiniLM, and Qwen inference runs through Modal.
- The UI log shows agent requests and responses but not authorization headers
  or tokens. It is meant for local use; the app binds to `127.0.0.1` by default.
- No Git repository, remote, commit, or upload is created by these scripts.
