"""Offline checks for planner orchestration and percentage-based selection."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from autonomous_agent import AutonomousPlanningAgent, PlanningState, SYSTEM_MESSAGE, USER_MESSAGE
from scanner.scanner_agent import Deal, DealSelection


def tool_call(name: str, arguments: dict, number: int) -> dict:
    return {
        "id": f"call_{number}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeResponse:
    def __init__(self, message: dict):
        self.message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": self.message}]}


class FakeSession:
    def __init__(self, messages: list[dict]):
        self.messages = iter(messages)
        self.requests: list[dict] = []

    def post(self, url, **kwargs):
        self.requests.append(kwargs["json"])
        return FakeResponse(next(self.messages))


class FakeScanner:
    def __init__(self, deals: list[Deal]):
        self.deals = deals
        self.calls = 0

    def scan(self):
        self.calls += 1
        return [object()] * 30, DealSelection(deals=self.deals)


class FakeEnsemble:
    def __init__(self, estimates: dict[str, int]):
        self.estimates = estimates
        self.calls: list[str] = []

    def predict_raw(self, description: str):
        self.calls.append(description)
        return {"ensemble_price_usd": self.estimates[description]}


class FakeMessenger:
    def __init__(self):
        self.calls: list[tuple] = []

    def announce(self, *args):
        self.calls.append(args)
        return f"Best deal: {args[0]} at ${args[2]} (expected ${args[1]}). {args[3]}"


class AutonomousAgentTests(unittest.TestCase):
    def setUp(self):
        self.deals = [
            Deal(
                product_description=f"Detailed product description number {index} with a useful feature",
                price=[100, 80, 200, 50, 150][index],
                url=f"https://example.com/deal-{index}",
            )
            for index in range(5)
        ]
        # Deal 2 has larger dollar savings; deal 1 wins on percentage.
        self.estimates = dict(zip(
            [deal.product_description for deal in self.deals],
            [150, 160, 300, 60, 200],
        ))
        self.environ = patch.dict(os.environ, {
            "MODAL_SHARED_ENDPOINT_URL": "https://example.modal.direct/v1/chat/completions",
            "MODAL_SHARED_ENDPOINT_MODEL": "Qwen/test",
            "MODAL_PROXY_TOKEN_ID": "test-id",
            "MODAL_PROXY_TOKEN_SECRET": "test-secret",
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def make_agent(self, messages):
        scanner = FakeScanner(self.deals)
        ensemble = FakeEnsemble(self.estimates)
        messenger = FakeMessenger()
        session = FakeSession(messages)
        agent = AutonomousPlanningAgent(
            scanner=scanner,
            ensemble_factory=lambda: ensemble,
            messaging_factory=lambda: messenger,
            session=session,
            snapshot_writer=lambda *_: Path("/tmp/fake-snapshot"),
        )
        return agent, scanner, ensemble, messenger, session

    def test_full_workflow_selects_highest_percentage_and_streams_steps(self):
        calls = [tool_call("scan_the_internet_for_bargains", {}, 0)]
        calls += [tool_call("estimate_true_value", {"url": str(deal.url)}, i + 1)
                  for i, deal in enumerate(self.deals)]
        calls += [tool_call("notify_user_of_deal", {}, 6)]
        messages = [{"tool_calls": [call]} for call in calls] + [{"content": "OK"}]
        agent, scanner, ensemble, messenger, session = self.make_agent(messages)

        states = list(agent.run_stream())
        final = states[-1]
        self.assertEqual(final["status"], "Complete")
        self.assertEqual(final["planner_reply"], "OK")
        self.assertEqual(final["best_url"], str(self.deals[1].url))
        self.assertEqual(len(final["comparisons"]), 5)
        self.assertEqual(final["comparisons"][1]["savings_pct"], 50.0)
        self.assertEqual(scanner.calls, 1)
        self.assertEqual(len(ensemble.calls), 5)
        self.assertEqual(len(messenger.calls), 1)
        self.assertEqual(messenger.calls[0][0], self.deals[1].product_description)
        self.assertEqual(session.requests[0]["messages"][:2], [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ])
        self.assertTrue(any(state["status"].startswith("Completed tool: scan") for state in states))
        log = final["agent_log"]
        self.assertEqual(log[0]["agent"], "Planning Agent (Qwen)")
        self.assertEqual(log[0]["direction"], "request")
        self.assertEqual(log[1]["direction"], "response")
        self.assertEqual(
            [(event["agent"], event["direction"]) for event in log if event["agent"] == "Scanner Agent"],
            [("Scanner Agent", "request"), ("Scanner Agent", "response")],
        )
        self.assertEqual(
            len([event for event in log if event["agent"] == "Ensemble Agent" and event["direction"] == "response"]),
            5,
        )
        self.assertEqual(
            [(event["agent"], event["direction"]) for event in log if event["agent"] == "Messaging Agent"],
            [("Messaging Agent", "request"), ("Messaging Agent", "response")],
        )
        self.assertNotIn("test-secret", json.dumps(log))
        scan_start = next(state for state in states if state["status"] == "Calling Scanner Agent")
        self.assertEqual(scan_start["comparisons"], [])

    def test_notification_requires_all_five_estimates(self):
        agent, _, _, messenger, _ = self.make_agent([])
        state = PlanningState(selected=self.deals)
        with self.assertRaisesRegex(ValueError, "all five"):
            agent._execute_tool("notify_user_of_deal", {}, state)
        self.assertEqual(messenger.calls, [])

    def test_only_scanned_urls_can_be_priced(self):
        agent, _, _, _, _ = self.make_agent([])
        state = PlanningState(selected=self.deals)
        with self.assertRaisesRegex(ValueError, "not one of"):
            agent._execute_tool("estimate_true_value", {"url": "https://example.com/other"}, state)


if __name__ == "__main__":
    unittest.main()
