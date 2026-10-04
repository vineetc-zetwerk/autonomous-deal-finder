"""No-cloud tests for the self-contained evaluator."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from plotly.graph_objects import Figure

from evaluation import evaluate


@dataclass
class Item:
    title: str
    price: float


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_two_figures(self):
        items = [Item("One", 10), Item("Two", 20), Item("Three", 30)]
        with patch.object(Figure, "show") as show:
            metrics = evaluate(lambda item: item.price + 2, items, size=3, workers=1)
        self.assertEqual(metrics["count"], 3)
        self.assertEqual(metrics["mae_usd"], 2)
        self.assertEqual(show.call_count, 2)

    def test_rejects_incomplete_input(self):
        with self.assertRaises(ValueError):
            evaluate(lambda item: item.price, [Item("One", 10)], size=2)


if __name__ == "__main__":
    unittest.main()
