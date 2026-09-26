"""A failed tool call must be distinguishable from a successful one.

`CallToolResponse.is_error` (schema.py, the MCP protocol's own error flag) was
read from the server and dropped on the next line. A failed call was persisted
byte-identically to a successful one, so nothing structured could answer "how
often did a tool call fail?".

Measured after the fact, by string-matching error text that happened to survive
in `content`: 2,336 of 10,915 tool calls failed in the 2026-08-29 grid (21.4%),
and the rate tracked the compared conditions inversely -- 12.4% under
`full_exposure`, 31.6% under `one_shot_b8`. A difference that large, aligned
that closely with the thing being compared, is not a detail worth losing.

The third test pins the property that makes the fix safe: the flag is a sibling
of `data` in the outputs record and never inside it, because `data` is also what
reaches the provider in `visible_messages`. Putting it inside would change what
the MODEL sees -- a behavioural change, not observability.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from mcp_completion.dynamic_eval import DynamicSelection, run_dynamic_mcp_eval
from mcp_completion.schema import CallToolResponse, TextContent, UserMessage

GIT_LOG_FAILURE = (
    '{"detail":"Failed to call tool \'git_git_log\': '
    "Reference at 'refs/heads/master' does not exist\"}"
)


@dataclass
class _ToolCall:
    id: str
    function: dict[str, Any]


@dataclass
class _AssistantMessage:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": c.function}
                for c in self.tool_calls or []
            ] or None,
        }


@dataclass
class _CompletionResult:
    message: _AssistantMessage


class _Client:
    """Raw MCP client whose single tool succeeds or fails, as configured."""

    def __init__(self, *, is_error: bool):
        self.is_error = is_error
        self.tools = [{"name": "git_git_log", "description": "log",
                       "inputSchema": {"type": "object"}}]

    async def list_raw_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> CallToolResponse:
        text = GIT_LOG_FAILURE if self.is_error else "result:ok"
        return CallToolResponse(content=[TextContent(type="text", text=text)],
                                isError=self.is_error)


class _Completion:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = [
            _CompletionResult(_AssistantMessage(tool_calls=[
                _ToolCall(id="c1", function={"name": "git_git_log", "arguments": "{}"})])),
            _CompletionResult(_AssistantMessage(content="done")),
        ]

    async def __call__(self, *, model, messages, tools, extra_body):
        self.requests.append({"messages": [m.model_dump() for m in messages]})
        return self._responses.pop(0)


@dataclass
class _Selector:
    calls: list[int] = field(default_factory=list)

    def select(self, *, raw_tools, visible_messages, cycle_index):
        self.calls.append(cycle_index)
        return DynamicSelection(active_tool_names=("git_git_log",))


async def _run(is_error: bool):
    completion = _Completion()
    result = await run_dynamic_mcp_eval(
        mcp_client=_Client(is_error=is_error),
        selector=_Selector(),
        completion=completion,
        model="fake/model",
        messages=[UserMessage(role="user", content="check the log")],
        max_turns=5,
    )
    return result, completion


class ToolFailureIsRecordedTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_tool_call_is_recorded_as_failed(self):
        result, _ = await _run(is_error=True)
        tool_entries = [o for o in result.outputs
                        if o.get("data", {}).get("role") == "tool"]
        self.assertTrue(tool_entries, "no tool result was recorded at all")
        self.assertIs(tool_entries[0]["tool_call_failed"], True)
        self.assertEqual(tool_entries[0]["tool_name"], "git_git_log")

    async def test_successful_tool_call_is_recorded_as_not_failed(self):
        result, _ = await _run(is_error=False)
        tool_entries = [o for o in result.outputs
                        if o.get("data", {}).get("role") == "tool"]
        self.assertTrue(tool_entries)
        self.assertIs(tool_entries[0]["tool_call_failed"], False)

    async def test_flag_never_reaches_the_model(self):
        """The provider payload must be byte-unchanged by this fix."""
        _, completion = await _run(is_error=True)
        tool_messages = [m for request in completion.requests
                         for m in request["messages"] if m.get("role") == "tool"]
        self.assertTrue(tool_messages, "the model never saw the tool result")
        for message in tool_messages:
            self.assertNotIn("tool_call_failed", message)
            self.assertNotIn("tool_name", message)


if __name__ == "__main__":
    unittest.main()
