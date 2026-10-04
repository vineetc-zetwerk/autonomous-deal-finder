"""Modal inference for canonical item descriptions and the QLoRA price model.

Run from the repository root:
    uv run modal run specialist/price_pipeline.py --text "A used ..."

The same canonical description returned by Preprocessor is used by the two
ensemble members. No model weights run on the local machine.
"""

from __future__ import annotations

import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import modal


QUESTION = "What does this cost to the nearest dollar?"
FIELDS = ("Title", "Category", "Brand", "Description", "Details")
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ADAPTER = "chitlangiavineet09/llama-3.2-3b-price-predictor-qlora"

# This file is read only by the local Modal client. Modal workers get HF_TOKEN
# through an ephemeral Modal Secret and never receive the .env file itself.
if modal.is_local():
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    if project_root.name == "myfiles" and project_root.parent.name == "week8":
        load_dotenv(project_root.parent / ".env", override=False)

BASE_MODEL = os.getenv("HF_BASE_MODEL_ID", DEFAULT_BASE_MODEL)
ADAPTER = os.getenv("HF_PRICE_ADAPTER_ID", DEFAULT_ADAPTER)
PREPROCESSOR_MODEL = os.getenv("PRICER_PREPROCESSOR_MODEL", DEFAULT_BASE_MODEL)

if BASE_MODEL != DEFAULT_BASE_MODEL:
    raise ValueError(
        f"This adapter declares {DEFAULT_BASE_MODEL} as its base model; "
        "set HF_BASE_MODEL_ID to that value."
    )

if modal.is_local():
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Add an HF_TOKEN with Llama access to the project .env.")
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": token})
else:
    hf_secret = modal.Secret.from_dict({})

app = modal.App("week8-price-pipeline")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.5.1",
    "transformers==4.56.2",
    "peft==0.20.0",
    "accelerate==1.10.1",
    "huggingface_hub==0.34.4",
)
cache = modal.Volume.from_name("week8-price-hf-cache", create_if_missing=True)
model_image = image.env({"HF_HOME": "/model-cache"})


def canonical_description(text: str) -> str:
    """Validate and normalize the five fields shared by every predictor."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Item details must be nonempty text.")
    values: dict[str, str] = {}
    current: str | None = None
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(Title|Category|Brand|Description|Details)(?::\s*(.*))?$", line)
        if match:
            current = match.group(1)
            if current in values:
                raise ValueError(f"Duplicate {current} field in item description.")
            values[current] = (match.group(2) or "").strip()
        elif current and (not values[current] or current in {"Description", "Details"}):
            values[current] = (values[current] + " " + line).strip()
        else:
            raise ValueError("Item description contains text outside the five labeled fields.")
    missing = [field for field in FIELDS if not values.get(field)]
    if missing:
        raise ValueError("Missing item fields: " + ", ".join(missing))
    return "\n".join(f"{field}: {values[field]}" for field in FIELDS)


def price_messages(description: str) -> list[dict[str, str]]:
    """Match the exact user-message shape in the uploaded training JSONL."""
    return [
        {
            "role": "user",
            "content": f"{QUESTION}\n\n{canonical_description(description)}",
        }
    ]


def parse_price(output: str) -> int:
    """Return a whole USD estimate; fail rather than invent a default price."""
    match = re.fullmatch(r"\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*", output)
    if not match:
        raise ValueError(f"Price model returned a nonnumeric response: {output!r}")
    try:
        amount = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation as error:
        raise ValueError("Price model returned an invalid amount.") from error
    return int(amount.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@app.cls(
    image=model_image,
    gpu="T4",
    secrets=[hf_secret],
    volumes={"/model-cache": cache},
    timeout=600,
)
class Preprocessor:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(PREPROCESSOR_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            PREPROCESSOR_MODEL, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    @modal.method()
    def normalize(self, item_details: str) -> str:
        import torch

        if not item_details.strip():
            raise ValueError("Item details must be nonempty text.")
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the item into exactly five labeled lines: Title, Category, "
                    "Brand, Description, Details. Put a colon after each label. "
                    "Preserve concrete product attributes. Copy an explicitly named "
                    "manufacturer into Brand; use 'Unknown' only when absent. "
                    "Omit prices, part numbers, "
                    "commentary, and markdown. Description and Details should each be "
                    "one concise sentence."
                ),
            },
            {"role": "user", "content": item_details},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=160, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        try:
            return canonical_description(generated)
        except ValueError as error:
            raise ValueError(f"Preprocessor output failed validation: {generated!r}") from error


@app.cls(
    image=model_image,
    gpu="T4",
    secrets=[hf_secret],
    volumes={"/model-cache": cache},
    timeout=600,
)
class Specialist:
    @modal.enter()
    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float16, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER)
        self.model.eval()

    @modal.method()
    def price(self, description: str) -> int:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            price_messages(description), tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=12, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return parse_price(answer)


@app.local_entrypoint()
def main(text: str) -> None:
    description = Preprocessor().normalize.remote(text)
    price = Specialist().price.remote(description)
    print("Canonical item description:\n" + description)
    print(f"Predicted price: ${price}")
