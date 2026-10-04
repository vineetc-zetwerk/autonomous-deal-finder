"""Validate the training-row boundary and RAG prompt without cloud calls."""

import unittest
from unittest.mock import Mock, patch

from local_store import open_local_collection, query_neighbors
from rag_pipeline import description_from_training_row, parse_rag_price, rag_messages
from shared_endpoint import price_from_neighbors


DESCRIPTION = """Title: HyperX QuadCast microphone
Category: Electronics
Brand: HyperX
Description: USB condenser microphone for clear audio.
Details: Connects to a computer by USB-C."""


class RagContractTests(unittest.TestCase):
    def test_training_row_excludes_question_and_price_from_document(self):
        row = {"messages": [
            {"role": "user", "content": "What does this cost to the nearest dollar?\n\n" + DESCRIPTION},
            {"role": "assistant", "content": "64.00"},
        ]}
        self.assertEqual(description_from_training_row(row), (DESCRIPTION, 64.0))

    def test_empty_training_brand_is_preserved(self):
        row = {"messages": [
            {"role": "user", "content": "What does this cost to the nearest dollar?\n\n" + DESCRIPTION.replace("Brand: HyperX", "Brand: ")},
            {"role": "assistant", "content": "64.00"},
        ]}
        document, price = description_from_training_row(row)
        self.assertIn("Brand: \n", document)
        self.assertEqual(price, 64.0)

    def test_prompt_has_five_reference_prices(self):
        neighbors = [{"description": DESCRIPTION, "price_usd": float(n)} for n in range(1, 6)]
        messages = rag_messages(DESCRIPTION, neighbors)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], (
            "Estimate the price of this product. Respond with the price, no explanation\n\n"
            + DESCRIPTION + "\n\n"
            "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
            + "\n\n".join(
                f"Reference {number} (known price ${number:.2f}):\n{DESCRIPTION}"
                for number in range(1, 6)
            )
        ))

    def test_final_price_parser(self):
        self.assertEqual(parse_rag_price("<think>compare examples</think>\n\n$64.50"), 65)
        self.assertEqual(parse_rag_price("**155.00**"), 155)
        with self.assertRaisesRegex(ValueError, "invalid price"):
            parse_rag_price("maybe sixty dollars")

    def test_endpoint_retry_keeps_exactly_one_user_message(self):
        neighbors = [{"description": DESCRIPTION, "price_usd": float(n)} for n in range(1, 6)]
        response = Mock()
        response.json.side_effect = [
            {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": "64.00"}, "finish_reason": "stop"}]},
        ]
        with patch.dict("os.environ", {
            "MODAL_PROXY_TOKEN_ID": "test-id",
            "MODAL_PROXY_TOKEN_SECRET": "test-secret",
        }), patch("shared_endpoint.requests.post", return_value=response) as post:
            result = price_from_neighbors(
                DESCRIPTION, neighbors,
                "https://example.modal.direct/v1/chat/completions", "Qwen/test",
            )
        self.assertEqual(result["price_usd"], 64)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["messages"], rag_messages(DESCRIPTION, neighbors))
        self.assertEqual(post.call_args_list[1].kwargs["json"]["messages"], rag_messages(DESCRIPTION, neighbors))

    def test_local_chroma_search_uses_precomputed_embedding(self):
        collection = open_local_collection()
        record = collection.get(ids=["train-00000"], include=["embeddings"])
        embedding = record["embeddings"][0].tolist()
        neighbors = query_neighbors(embedding)
        self.assertEqual(len(neighbors), 5)
        self.assertEqual(neighbors[0]["id"], "train-00000")
        self.assertEqual(neighbors[0]["price_usd"], 64.0)


if __name__ == "__main__":
    unittest.main()
