"""Draft a deal announcement with Qwen on the existing Modal Shared Endpoint.

From this project's root:
    uv run python messaging/messaging_agent.py \
        --description "USB condenser microphone with a desktop stand" \
        --expected-price 150 --deal-price 99 \
        --url "https://example.com/deal"

This agent only returns text. It does not send a notification or message.
"""

from __future__ import annotations

import argparse
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError


SYSTEM_PROMPT = (
    "Write one short, appealing but factual product sentence for a deal announcement. "
    "Mention only product attributes explicitly stated in the description. "
    "Do not include prices, savings, links, calls to action, or unsupported claims. "
    "Return only the requested JSON object."
)


class ProductHook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_hook: str = Field(min_length=10, max_length=180)


def _price(value: str | int | float | Decimal, name: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a positive USD amount.") from error
    if not amount.is_finite() or amount <= 0 or amount.as_tuple().exponent < -2:
        raise ValueError(f"{name} must be a positive USD amount with at most two decimal places.")
    return amount


def _money(amount: Decimal) -> str:
    return f"${amount:,.2f}".removesuffix(".00")


def validate_inputs(
    product_description: str,
    expected_price: str | int | float | Decimal,
    deal_price: str | int | float | Decimal,
    deal_page_url: str,
) -> tuple[str, Decimal, Decimal, str]:
    description = product_description.strip()
    if len(description) < 10:
        raise ValueError("product_description must contain at least 10 characters.")
    expected = _price(expected_price, "expected_price")
    deal = _price(deal_price, "deal_price")
    if deal >= expected:
        raise ValueError("deal_price must be lower than expected_price for a deal announcement.")
    url = deal_page_url.strip()
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or re.search(r"\s", url)
    ):
        raise ValueError("deal_page_url must be a valid HTTPS URL without embedded credentials.")
    return description, expected, deal, url


class MessagingAgent:
    """Use Qwen for product wording; calculate and attach financial facts locally."""

    name = "Messaging Agent"

    def __init__(self, session: requests.Session | None = None):
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env", override=False)
        if project_root.name == "myfiles" and project_root.parent.name == "week8":
            load_dotenv(project_root.parent / ".env", override=False)
        self.session = session or requests.Session()

    def announce(
        self,
        product_description: str,
        expected_price: str | int | float | Decimal,
        deal_price: str | int | float | Decimal,
        deal_page_url: str,
    ) -> str:
        description, expected, deal, url = validate_inputs(
            product_description, expected_price, deal_price, deal_page_url
        )
        endpoint_url = os.getenv("MODAL_SHARED_ENDPOINT_URL", "").strip()
        endpoint_model = os.getenv("MODAL_SHARED_ENDPOINT_MODEL", "").strip()
        proxy_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
        proxy_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
        parsed = urlparse(endpoint_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".modal.direct")
            or parsed.path != "/v1/chat/completions"
        ):
            raise RuntimeError("Set a valid Modal Shared Endpoint URL in the project .env.")
        if not endpoint_model or not proxy_id or not proxy_secret:
            raise RuntimeError("Set the Modal endpoint model and proxy token in the project .env.")

        response = self.session.post(
            endpoint_url,
            headers={
                "Authorization": f"Bearer {proxy_id}.{proxy_secret}",
                "Content-Type": "application/json",
            },
            json={
                "model": endpoint_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Product description:\n{description}"},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "product_hook",
                        "strict": True,
                        "schema": ProductHook.model_json_schema(),
                    },
                },
                "reasoning_effort": "low",
                "max_tokens": 512,
                "temperature": 0,
                "stream": False,
            },
            timeout=240,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        if not content:
            raise ValueError(
                "Qwen returned no announcement content "
                f"(finish_reason={choice.get('finish_reason')!r})."
            )
        try:
            hook = ProductHook.model_validate_json(content).product_hook.strip()
        except ValidationError as error:
            raise ValueError("Qwen announcement did not match the Pydantic schema.") from error
        if "$" in hook or re.search(r"https?://", hook, flags=re.IGNORECASE):
            raise ValueError("Qwen product sentence unexpectedly included a price or URL.")
        hook = hook.rstrip(" .!?")
        return (
            f"Deal: {hook}. Expected price: {_money(expected)}; "
            f"deal price: {_money(deal)}; save {_money(expected - deal)}. {url}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", required=True, help="Product description")
    parser.add_argument("--expected-price", required=True, help="Expected USD price")
    parser.add_argument("--deal-price", required=True, help="Deal USD price")
    parser.add_argument("--url", required=True, help="HTTPS deal page URL")
    args = parser.parse_args()
    announcement = MessagingAgent().announce(
        args.description, args.expected_price, args.deal_price, args.url
    )
    print(announcement)


if __name__ == "__main__":
    main()
