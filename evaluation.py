"""Evaluate price predictions without depending on another repository.

``Figure.show`` is used for both charts so the ensemble runner can save them
as HTML without opening a browser.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate

import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(predictor, data, size: int = 200, workers: int = 5) -> dict:
    if size < 1 or workers < 1 or len(data) < size:
        raise ValueError("Evaluation needs at least size items and positive size/workers.")

    items = data[:size]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        guesses = [float(value) for value in executor.map(predictor, items)]
    truths = [float(item.price) for item in items]
    errors = [abs(guess - truth) for guess, truth in zip(guesses, truths)]
    running_means = [total / count for count, total in enumerate(accumulate(errors), 1)]

    trend = go.Figure()
    trend.add_trace(go.Scatter(
        x=list(range(1, size + 1)), y=running_means, mode="lines",
        name="Running mean absolute error",
    ))
    trend.update_layout(
        title="Mean absolute error across test items",
        xaxis_title="Items evaluated", yaxis_title="Mean absolute error (USD)",
    )
    trend.show()

    maximum = max([*truths, *guesses])
    comparison = go.Figure()
    comparison.add_trace(go.Scatter(
        x=truths, y=guesses, mode="markers",
        text=[item.title for item in items], name="Predictions",
        hovertemplate="%{text}<br>Actual $%{x:,.2f}<br>Predicted $%{y:,.2f}<extra></extra>",
    ))
    comparison.add_trace(go.Scatter(
        x=[0, maximum], y=[0, maximum], mode="lines", name="Perfect prediction",
        line={"dash": "dash"},
    ))
    comparison.update_layout(
        title="Actual versus predicted price",
        xaxis_title="Actual price (USD)", yaxis_title="Predicted price (USD)",
    )
    comparison.show()

    metrics = {
        "count": size,
        "mae_usd": float(mean_absolute_error(truths, guesses)),
        "mse_usd_squared": float(mean_squared_error(truths, guesses)),
        "r2": float(r2_score(truths, guesses)) if size > 1 else None,
    }
    print(metrics, flush=True)
    return metrics
