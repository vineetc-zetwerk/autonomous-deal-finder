"""Gradio control panel for the on-demand autonomous bargain planner.

From this project's root:
    uv run python autonomous/app.py
"""

from __future__ import annotations

import html
import json
import os

import gradio as gr

from autonomous_agent import AutonomousPlanningAgent


def display(state: dict) -> tuple[str, str, list[list], str, str, str]:
    rows = [
        [
            item["product"],
            item["deal_price_usd"],
            item["expected_price_usd"],
            round(item["savings_pct"], 1) if item["savings_pct"] is not None else None,
        ]
        for item in state["comparisons"]
    ]
    deal_link = ""
    if state["best_url"]:
        safe_url = html.escape(state["best_url"], quote=True)
        deal_link = (
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">'
            "Open winning deal</a>"
        )
    log = "\n\n".join(
        f'[{event["time"]}] {event["agent"]} — {event["direction"].upper()}\n'
        + json.dumps(event["payload"], indent=2, ensure_ascii=False, default=str)
        for event in state["agent_log"]
    )
    return (
        state["status"],
        "\n".join(f"{number}. {step}" for number, step in enumerate(state["steps"], 1)),
        rows,
        state["announcement"] or "",
        deal_link,
        log,
    )


def run_planner():
    last_state = {
        "status": "Starting",
        "steps": [],
        "comparisons": [],
        "announcement": None,
        "best_url": None,
        "agent_log": [],
    }
    yield display(last_state)
    try:
        agent = AutonomousPlanningAgent()
        for state in agent.run_stream():
            last_state = state
            yield display(state)
    except Exception as error:
        # Preserve completed comparisons and make a failed cloud/feed step visible.
        last_state["status"] = "Run failed"
        last_state["steps"].append(f"Error: {type(error).__name__}: {error}")
        yield display(last_state)


with gr.Blocks(title="Autonomous Deal Finder") as app:
    gr.Markdown(
        "# Autonomous Deal Finder\n"
        "Scan live RSS deals, estimate five prices with the ensemble, and show the "
        "deal with the largest positive percentage saving. Nothing is sent externally."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=480):
            trigger = gr.Button("Find the best deal now", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            steps = gr.Textbox(label="Agent steps", lines=11, interactive=False)
            comparisons = gr.Dataframe(
                headers=["Product", "Deal price (USD)", "Expected price (USD)", "Savings (%)"],
                datatype=["str", "number", "number", "number"],
                interactive=False,
                label="Five scanned deals and ensemble estimates",
            )
            announcement = gr.Textbox(label="Best deal announcement", lines=4, interactive=False)
            link = gr.HTML()
        with gr.Column(scale=1, min_width=480):
            gr.Markdown("### Live agent log\nRequests and responses; credentials are not shown.")
            agent_log = gr.Textbox(label="Agent calls", lines=32, interactive=False,
                                   show_copy_button=True)
    trigger.click(
        run_planner,
        outputs=[status, steps, comparisons, announcement, link, agent_log],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        inbrowser=False,
    )
