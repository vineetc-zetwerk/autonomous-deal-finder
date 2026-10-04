"""Call the Modal Qwen Shared Endpoint with locally retrieved RAG context."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests

if __package__:
    from .rag_pipeline import parse_rag_price, rag_messages
else:
    from rag_pipeline import parse_rag_price, rag_messages


def price_from_neighbors(
    description: str, neighbors: list[dict], endpoint_url: str, endpoint_model: str
) -> dict:
    """The HTTP client is local; all LLM inference runs on Modal's endpoint."""
    parsed_url = urlparse(endpoint_url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or not parsed_url.hostname.endswith(".modal.direct")
        or parsed_url.path != "/v1/chat/completions"
    ):
        raise ValueError("Expected a Modal HTTPS chat-completions endpoint URL.")
    if not endpoint_model.strip():
        raise ValueError("A Modal Shared Endpoint model ID is required.")
    proxy_id = os.getenv("MODAL_PROXY_TOKEN_ID", "").strip()
    proxy_secret = os.getenv("MODAL_PROXY_TOKEN_SECRET", "").strip()
    if not proxy_id or not proxy_secret:
        raise RuntimeError("Modal proxy token is missing from the project .env.")

    messages = rag_messages(description, neighbors)
    last_failure = ""
    for attempt, max_tokens in enumerate((1024, 4096, 4096)):
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {proxy_id}.{proxy_secret}"},
            json={
                "model": endpoint_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "reasoning_effort": "low",
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        answer = choice["message"].get("content") or ""
        try:
            return {"price_usd": parse_rag_price(answer), "neighbors": neighbors}
        except ValueError as error:
            last_failure = (
                f"attempt {attempt + 1}: {error}; "
                f"finish_reason={choice.get('finish_reason')!r}; "
                f"usage={payload.get('usage')!r}"
            )
            # Retry with more output tokens, but keep the user's single-message
            # prompt unchanged on every attempt.
    raise ValueError("Qwen did not return a usable price after retries: " + last_failure)
