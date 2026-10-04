"""No-network contract tests for the RSS scanner and structured selection."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from scanner_agent import (
    DealSelection, RSSDeal, ScannerAgent, clean_html, price_evidence,
    validate_selection,
)


def sample_deals() -> list[RSSDeal]:
    return [
        RSSDeal(
            "https://example.com/rss",
            f"Detailed product {number} for ${number + 10}",
            "An item with a precise product name, model, features, and free shipping.",
            f"https://www.dealnews.com/item-{number}.html",
        )
        for number in range(30)
    ]


class ScannerAgentTests(unittest.TestCase):
    def test_html_cleanup_and_discount_exclusion(self):
        self.assertEqual(clean_html("<div>Fast <b>laptop</b>  &amp; charger</div>"), "Fast laptop & charger")
        self.assertEqual(price_evidence("for $111, $166 off the $277 list price"), {111})
        self.assertEqual(price_evidence("save $100; reduced by $50; up to $300 off"), set())

    def test_prompt_uses_all_thirty_deals(self):
        prompt = ScannerAgent.make_user_prompt(sample_deals())
        self.assertEqual(prompt.count("Title: Detailed product"), 30)
        self.assertTrue(prompt.endswith("Include exactly 5 deals, no more."))

    def test_pydantic_requires_five_positive_deals(self):
        with self.assertRaises(ValueError):
            DealSelection.model_validate({"deals": []})
        with self.assertRaises(ValueError):
            DealSelection.model_validate({"deals": [{
                "product_description": "A sufficiently detailed product description for a test.",
                "price": 0, "url": "https://www.dealnews.com/item.html",
            }] * 5})

    def test_selected_price_must_be_in_the_source(self):
        deals = sample_deals()
        selection = DealSelection.model_validate({"deals": [
            {
                "product_description": "A detailed product summary about model features and construction.",
                "price": number + 10,
                "url": deals[number].url,
            }
            for number in range(5)
        ]})
        validate_selection(selection, deals)
        selection.deals[0].price = 999
        with self.assertRaisesRegex(ValueError, "No clear stated price"):
            validate_selection(selection, deals)

    def test_endpoint_uses_json_schema_and_given_prompts(self):
        deals = sample_deals()
        payload = {"deals": [
            {
                "product_description": "A detailed product summary about model features and construction.",
                "price": number + 10,
                "url": deals[number].url,
            }
            for number in range(5)
        ]}
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
        session = Mock()
        session.post.return_value = response
        with patch.dict(os.environ, {
            "MODAL_SHARED_ENDPOINT_URL": "https://example.modal.direct/v1/chat/completions",
            "MODAL_SHARED_ENDPOINT_MODEL": "Qwen/test",
            "MODAL_PROXY_TOKEN_ID": "test-id",
            "MODAL_PROXY_TOKEN_SECRET": "test-secret",
        }):
            result = ScannerAgent(session).select_deals(deals)
        self.assertEqual(len(result.deals), 5)
        request = session.post.call_args.kwargs["json"]
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        self.assertEqual(request["messages"][1]["content"].count("Title: Detailed product"), 30)


if __name__ == "__main__":
    unittest.main()
