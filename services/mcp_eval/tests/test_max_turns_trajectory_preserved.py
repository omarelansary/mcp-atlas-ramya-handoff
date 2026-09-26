"""A turn-capped run must keep its trajectory (thesis defect P1029-D1).

The dynamic loop used to raise a bare ``DynamicMcpEvalError`` when ``max_turns``
was reached, and the accumulated ``outputs`` were discarded with the exception.
P1-029 recorded 8 turn-capped runs as an empty payload: no messages, no cycles,
no tool calls, no reasoning. The one question those runs exist to answer -- what
were 50 turns doing? -- had no evidence left to answer it. Five of the eight
were a single task, which is exactly the case worth reading.

This is OBSERVABILITY ONLY. The tests below pin both halves of that claim:
what must now be preserved, and what must NOT have changed. Specifically the
run is still a failure, still HTTP 500, still ``max_turns_exhausted``, and the
model still sees exactly what it saw before.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from mcp_completion.dynamic_eval import (
    DynamicMcpEvalError,
    DynamicSelection,
    MaxTurnsExhaustedError,
    run_dynamic_mcp_eval,
)
from mcp_completion.schema import CallToolResponse, TextContent, UserMessage

MAX_TURNS = 4


class _ToolCall:
    def __init__(self, call_id: str) -> None:
        self.id = call_id
        self.function = {"name": "git_git_log", "arguments": "{}"}


class _Message:
    """Minimal stand-in for the provider message object."""

    def __init__(self, *, tool_calls=None, content=None) -> None:
        self.role = "assistant"
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content,
                "tool_calls": [{"id": c.id, "function": c.function}
                               for c in (self.tool_calls or [])]}


@dataclass
class _Completion:
    """Never stops: always asks for another tool call, so the cap is reached."""

    finish_after: int | None = None
    requests: list[dict] = field(default_factory=list)
    calls: int = 0

    async def __call__(self, *, model, messages, tools, extra_body):
        self.requests.append({"messages": [m.model_dump() for m in messages]})
        self.calls += 1
        if self.finish_after is not None and self.calls >= self.finish_after:
            return _Result(_Message(content="done"))
        return _Result(_Message(tool_calls=[_ToolCall(f"call-{self.calls}")]))


@dataclass
class _Result:
    original_message: Any
    finish_reason: str | None = "tool_calls"
    usage: dict | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        self.message = self.original_message


class _Client:
    def __init__(self) -> None:
        self.tools = [{"name": "git_git_log", "description": "log",
                       "inputSchema": {"type": "object"}}]

    async def list_raw_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict) -> CallToolResponse:
        return CallToolResponse(
            content=[TextContent(type="text", text="commit abc123")],
            isError=False)


class _Selector:
    def select(self, *, raw_tools, visible_messages, cycle_index):
        return DynamicSelection(active_tool_names=("git_git_log",))


async def _run(completion: _Completion):
    return await run_dynamic_mcp_eval(
        mcp_client=_Client(), selector=_Selector(), completion=completion,
        model="fake/model",
        messages=[UserMessage(role="user", content="read the log")],
        max_turns=MAX_TURNS,
    )


class MaxTurnsTrajectoryTests(unittest.IsolatedAsyncioTestCase):

    async def test_partial_trajectory_is_preserved_on_turn_cap(self):
        completion = _Completion()
        with self.assertRaises(MaxTurnsExhaustedError) as caught:
            await _run(completion)
        error = caught.exception

        self.assertTrue(error.outputs, "the trajectory was lost -- P1029-D1")
        tool_results = [o for o in error.outputs
                        if (o.get("data") or {}).get("role") == "tool"]
        assistant = [o for o in error.outputs
                     if (o.get("data") or {}).get("role") == "assistant"]
        self.assertEqual(len(assistant), MAX_TURNS)
        self.assertEqual(len(tool_results), MAX_TURNS)
        self.assertEqual(len(error.cycles), MAX_TURNS)
        # The flag that made the tool-failure census possible must survive too.
        self.assertIn("tool_call_failed", tool_results[0])

    async def test_turn_cap_is_still_a_failure(self):
        """Observability only: this must NOT become a successful run."""
        with self.assertRaises(MaxTurnsExhaustedError) as caught:
            await _run(_Completion())
        # The failure_code mapping in main.py matches on this exact string.
        self.assertEqual(str(caught.exception),
                         "model did not finish within max_turns")
        # Still a DynamicMcpEvalError, so every existing handler still catches it.
        self.assertIsInstance(caught.exception, DynamicMcpEvalError)

    async def test_model_visible_messages_are_unchanged(self):
        """The model must see exactly what it saw before the fix."""
        completion = _Completion()
        with self.assertRaises(MaxTurnsExhaustedError):
            await _run(completion)
        for request in completion.requests:
            for message in request["messages"]:
                self.assertNotIn("tool_call_failed", message)
                self.assertNotIn("tool_name", message)
                self.assertNotIn("partial_trajectory", message)
                self.assertNotIn("truncated_by", message)

    async def test_normal_runs_are_unchanged(self):
        """A run that finishes before the cap still returns normally."""
        completion = _Completion(finish_after=2)
        result = await _run(completion)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(len(result.cycles), 2)
        self.assertTrue(result.outputs)

    async def test_stopping_point_is_unchanged(self):
        """The loop does exactly max_turns cycles, as it did before."""
        completion = _Completion()
        with self.assertRaises(MaxTurnsExhaustedError):
            await _run(completion)
        self.assertEqual(completion.calls, MAX_TURNS)


if __name__ == "__main__":
    unittest.main()
