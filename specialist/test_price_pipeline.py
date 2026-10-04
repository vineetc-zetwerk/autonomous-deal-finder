"""Small contract checks for the shared item format and price parser."""

import unittest

from price_pipeline import QUESTION, canonical_description, parse_price, price_messages


DESCRIPTION = """Title: Schlage interior knob
Category: Home Hardware
Brand: Schlage
Description: Bronze interior door knob.
Details: Includes deadbolt hardware."""


class PricePipelineContractTests(unittest.TestCase):
    def test_training_prompt_shape(self):
        self.assertEqual(
            price_messages(DESCRIPTION),
            [{"role": "user", "content": QUESTION + "\n\n" + DESCRIPTION}],
        )

    def test_canonical_format_requires_every_field(self):
        with self.assertRaisesRegex(ValueError, "Missing item fields: Details"):
            canonical_description(DESCRIPTION.replace("Details: Includes deadbolt hardware.", ""))

    def test_standalone_labels_are_normalized(self):
        standalone = DESCRIPTION.replace(": ", "\n")
        self.assertEqual(canonical_description(standalone), DESCRIPTION)

    def test_price_is_rounded_to_whole_usd(self):
        self.assertEqual(parse_price("64.00"), 64)
        self.assertEqual(parse_price("$64.50"), 65)
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            parse_price("approximately 64 dollars")


if __name__ == "__main__":
    unittest.main()
