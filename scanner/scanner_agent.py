"""Scan 30 RSS deals, then select five with a Modal-hosted Qwen endpoint.

From this project's root:
    uv run python scanner/scanner_agent.py --rss-only
    uv run python scanner/scanner_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError


FEEDS = (
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
)
ENTRIES_PER_FEED = 10
EXPECTED_DEALS = len(FEEDS) * ENTRIES_PER_FEED
OUTPUT_ROOT = Path(__file__).resolve().parent / "scanner_outputs"

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.
"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."


class Deal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_description: str = Field(min_length=30)
    price: float = Field(gt=0, allow_inf_nan=False)
    url: HttpUrl


class DealSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deals: list[Deal] = Field(min_length=5, max_length=5)


@dataclass(frozen=True)
class RSSDeal:
    source_feed: str
    title: str
    details: str
    url: str

    def prompt_text(self) -> str:
        return f"Title: {self.title}\nDetails: {self.details}\nURL: {self.url}"

    def as_dict(self) -> dict[str, str]:
        return {
            "source_feed": self.source_feed,
            "title": self.title,
            "details": self.details,
            "url": self.url,
        }


def clean_html(value: str) -> str:
    text = BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def price_evidence(text: str) -> set[Decimal]:
    """Find stated USD amounts, excluding obvious discount amounts."""
    values: set[Decimal] = set()
    for match in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\b", text):
        before = text[max(0, match.start() - 25):match.start()].lower()
        after = text[match.end():match.end() + 15].lower()
        if re.search(
            r"(?:save|savings of|reduced by|discount(?: of)?|up to|list price|regular(?: price)?|original(?: price)?|was|normally)\s*$",
            before,
        ):
            continue
        if re.match(r"\s*(?:off\b|discount\b|list price\b|msrp\b)", after):
            continue
        try:
            amount = Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        if amount > 0:
            values.add(amount)
    return values


def validate_selection(selection: DealSelection, scraped: list[RSSDeal]) -> None:
    by_url = {deal.url: deal for deal in scraped}
    seen: set[str] = set()
    for selected in selection.deals:
        url = str(selected.url)
        if url not in by_url or url in seen:
            raise ValueError(f"Selected URL is absent from the feeds or duplicated: {url}")
        seen.add(url)
        source = by_url[url]
        evidence = price_evidence(source.title + " " + source.details)
        stated_price = Decimal(str(selected.price))
        if not any(abs(stated_price - amount) <= Decimal("0.01") for amount in evidence):
            raise ValueError(
                f"No clear stated price ${selected.price:g} for {url}; "
                "the selected amount may be a discount or an invention."
            )


class ScannerAgent:
    """RSS parsing is local and deterministic; only selection uses an LLM."""

    name = "Scanner Agent"

    def __init__(self, session: requests.Session | None = None):
        project_root = Path(__file__).resolve().parents[1]
        load_dotenv(project_root / ".env", override=False)
        if project_root.name == "myfiles" and project_root.parent.name == "week8":
            load_dotenv(project_root.parent / ".env", override=False)
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; Week8Scanner/1.0)",
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        })

    def fetch_deals(self) -> list[RSSDeal]:
        deals: list[RSSDeal] = []
        for feed_url in FEEDS:
            response = self.session.get(feed_url, timeout=30)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            if len(parsed.entries) < ENTRIES_PER_FEED:
                raise RuntimeError(
                    f"Expected {ENTRIES_PER_FEED} RSS entries from {feed_url}, "
                    f"found {len(parsed.entries)}."
                )
            for entry in parsed.entries[:ENTRIES_PER_FEED]:
                title = clean_html(entry.get("title", ""))
                details = clean_html(entry.get("summary", ""))
                url = entry.get("link", "").strip()
                if not title or not details or not url.startswith("https://"):
                    raise RuntimeError(f"Incomplete RSS entry in {feed_url}.")
                deals.append(RSSDeal(feed_url, title, details, url))
        if len(deals) != EXPECTED_DEALS:
            raise RuntimeError(f"Expected {EXPECTED_DEALS} RSS deals, got {len(deals)}.")
        return deals

    @staticmethod
    def make_user_prompt(deals: list[RSSDeal]) -> str:
        if len(deals) != EXPECTED_DEALS:
            raise ValueError(f"Expected {EXPECTED_DEALS} deals before calling Qwen.")
        return USER_PROMPT_PREFIX + "\n\n".join(deal.prompt_text() for deal in deals) + USER_PROMPT_SUFFIX

    def select_deals(self, deals: list[RSSDeal]) -> DealSelection:
        if len(deals) != EXPECTED_DEALS:
            raise ValueError(f"Expected {EXPECTED_DEALS} RSS deals.")
        if sum(bool(price_evidence(deal.title + " " + deal.details)) for deal in deals) < 5:
            raise ValueError("Fewer than five RSS deals show a clear positive price; selection aborted.")

        endpoint_url = os.getenv("MODAL_SHARED_ENDPOINT_URL", "").strip()
        endpoint_model = os.getenv("MODAL_SHARED_ENDPOINT_MODEL", "").strip()
        proxy_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
        proxy_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
        parsed_url = urlparse(endpoint_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or not parsed_url.hostname.endswith(".modal.direct")
            or parsed_url.path != "/v1/chat/completions"
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
                    {"role": "user", "content": self.make_user_prompt(deals)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "deal_selection",
                        "strict": True,
                        "schema": DealSelection.model_json_schema(),
                    },
                },
                "reasoning_effort": "low",
                "max_tokens": 4096,
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
                "Qwen returned no structured content "
                f"(finish_reason={choice.get('finish_reason')!r})."
            )
        try:
            selection = DealSelection.model_validate_json(content)
        except ValidationError as error:
            raise ValueError("Qwen output did not match the five-deal Pydantic schema.") from error
        validate_selection(selection, deals)
        return selection

    def scan(self) -> tuple[list[RSSDeal], DealSelection]:
        deals = self.fetch_deals()
        return deals, self.select_deals(deals)


def save_snapshot(deals: list[RSSDeal], selection: DealSelection | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = OUTPUT_ROOT / stamp
    destination.mkdir(parents=True)
    (destination / "rss_deals.json").write_text(
        json.dumps([deal.as_dict() for deal in deals], indent=2, ensure_ascii=False) + "\n"
    )
    if selection is not None:
        (destination / "selected_deals.json").write_text(
            selection.model_dump_json(indent=2) + "\n"
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rss-only", action="store_true", help="Parse and save 30 RSS entries without LLM inference")
    args = parser.parse_args()
    agent = ScannerAgent()
    deals = agent.fetch_deals()
    counts = {feed: sum(deal.source_feed == feed for deal in deals) for feed in FEEDS}
    print("RSS entries by feed:", json.dumps(counts, indent=2))
    selection = None if args.rss_only else agent.select_deals(deals)
    destination = save_snapshot(deals, selection)
    print(f"Saved RSS snapshot: {destination / 'rss_deals.json'}")
    if selection is not None:
        print(selection.model_dump_json(indent=2))
        print(f"Saved selected deals: {destination / 'selected_deals.json'}")


if __name__ == "__main__":
    main()
