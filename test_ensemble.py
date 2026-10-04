"""Local, no-cloud checks for the two-model evaluation contract."""

import unittest

from ensemble import combine_prices, parse_test_row


class EnsembleContractTests(unittest.TestCase):
    def test_weighted_average_is_rounded_to_nearest_usd(self):
        self.assertEqual(combine_prices(100, 120), 116)
        self.assertEqual(combine_prices(100, 103), 102)

    def test_test_label_is_separate_from_model_input(self):
        row = {
            "row_index": 0,
            "expected_price": "115.99",
            "messages": [{"role": "user", "content": (
                "What does this cost to the nearest dollar?\n\n"
                "Title: Headlight Assembly\nCategory: Auto\nBrand: Telpo\n"
                "Description: Replacement headlight.\nDetails: Fits 2015-2017 Camry."
            )}],
        }
        item = parse_test_row(row, 0)
        self.assertEqual(item.title, "Headlight Assembly")
        self.assertEqual(item.price, 115.99)
        self.assertNotIn("115.99", item.description)


if __name__ == "__main__":
    unittest.main()
