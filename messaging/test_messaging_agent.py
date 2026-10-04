"""Offline tests for MessagingAgent; no Modal call is made."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from messaging_agent import MessagingAgent, validate_inputs


class FakeResponse:
    def __init__(self, content: str):
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]}


class FakeSession:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.content)


class MessagingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(
            "os.environ",
            {
                "MODAL_SHARED_ENDPOINT_URL": "https://example--endpoint.modal.direct/v1/chat/completions",
                "MODAL_SHARED_ENDPOINT_MODEL": "Qwen/example",
                "MODAL_PROXY_TOKEN_ID": "test-id",
                "MODAL_PROXY_TOKEN_SECRET": "test-secret",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_announcement_contains_exact_facts_and_supplied_link(self) -> None:
        session = FakeSession('{"product_hook":"A USB condenser microphone for desktop recording."}')
        message = MessagingAgent(session=session).announce(
            "USB condenser microphone for desktop recording",
            "150.00",
            "99.95",
            "https://example.com/deal?item=mic",
        )
        self.assertIn("Expected price: $150", message)
        self.assertIn("deal price: $99.95", message)
        self.assertIn("save $50.05", message)
        self.assertTrue(message.endswith("https://example.com/deal?item=mic"))
        request = session.calls[0]["json"]
        self.assertEqual(request["model"], "Qwen/example")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["messages"][1]["content"],
                         "Product description:\nUSB condenser microphone for desktop recording")

    def test_non_saving_deal_rejected_before_llm(self) -> None:
        session = FakeSession('{"product_hook":"A nice microphone"}')
        with self.assertRaisesRegex(ValueError, "lower than expected_price"):
            MessagingAgent(session=session).announce("A USB microphone", 100, 100, "https://example.com")
        self.assertEqual(session.calls, [])

    def test_bad_prices_and_urls_rejected(self) -> None:
        for price in ("0", "NaN", "Infinity", "1.001", "-1"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                validate_inputs("A USB microphone", 100, price, "https://example.com")
        for url in ("http://example.com", "https://user:pass@example.com", "https://example.com/a b"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_inputs("A USB microphone", 100, 50, url)

    def test_untrusted_llm_prices_and_links_are_rejected(self) -> None:
        for hook in (
            "A USB microphone costs $5 today",
            "A USB microphone is at https://other.example/deal",
        ):
            with self.subTest(hook=hook):
                import json
                session = FakeSession(json.dumps({"product_hook": hook}))
                with self.assertRaisesRegex(ValueError, "included a price or URL"):
                    MessagingAgent(session=session).announce(
                        "A USB microphone", 100, 50, "https://example.com/deal"
                    )

    def test_malformed_llm_response_rejected(self) -> None:
        session = FakeSession('{"wrong_key":"A USB microphone"}')
        with self.assertRaisesRegex(ValueError, "Pydantic schema"):
            MessagingAgent(session=session).announce(
                "A USB microphone", 100, 50, "https://example.com/deal"
            )


if __name__ == "__main__":
    unittest.main()
