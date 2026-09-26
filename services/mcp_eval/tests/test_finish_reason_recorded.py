"""A truncated turn must be distinguishable from one that finished.

litellm keeps `finish_reason` on `choices[0]`, NOT on the message -- `Message`
has no such field (content, role, tool_calls, function_call, audio, image,
reasoning_content, thinking_blocks, provider_specific_fields, annotations). So
code reading `original_message.finish_reason` gets `None` every time and will
report "0 truncated turns" whether or not anything truncated: an absent value
published as a measured zero.

That matters most under high reasoning effort with `max_tokens` unset, where a
cut-off answer scores lower on claim coverage and nothing in the record says
why.

The flag is a sibling of `data`, never inside it, for the same reason as
`tool_call_failed`: `data` is also what reaches the provider in
`visible_messages`, so a field added there would change what the MODEL sees.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from mcp_completion.dynamic_eval import DynamicSelection, run_dynamic_mcp_eval
from mcp_completion.schema import UserMessage

TOOL = {"name": "t", "description": "d", "inputSchema": {"type": "object"}}


@dataclass
class _AssistantMessage:
    content: str | None = None
    tool_calls: list | None = None

    def model_dump(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self.content, "tool_calls": None}


@dataclass
class _CompletionResult:
    message: _AssistantMessage
    finish_reason: str | None = None


class _Client:
    async def list_raw_tools(self):
        return [dict(TOOL)]


class _Completion:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        self.requests: list = []

    async def __call__(self, *, model, messages, tools, extra_body):
        self.requests.append([m.model_dump() for m in messages])
        return _CompletionResult(_AssistantMessage(content="done"),
                                 finish_reason=self.finish_reason)


@dataclass
class _Selector:
    selector_id: str = "test"
    calls: list = field(default_factory=list)

    def select(self, *, raw_tools, visible_messages, cycle_index):
        self.calls.append(cycle_index)
        return DynamicSelection(active_tool_names=("t",))


async def _run(finish_reason):
    completion = _Completion(finish_reason)
    result = await run_dynamic_mcp_eval(
        mcp_client=_Client(), selector=_Selector(), completion=completion,
        model="fake/model",
        messages=[UserMessage(role="user", content="go")], max_turns=3)
    return result, completion


class FinishReasonRecordedTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncation_is_recorded(self):
        result, _ = await _run("length")
        assistant = [o for o in result.outputs
                     if o.get("data", {}).get("role") == "assistant"]
        self.assertTrue(assistant)
        self.assertEqual(assistant[0]["finish_reason"], "length")

    async def test_normal_stop_is_recorded(self):
        result, _ = await _run("stop")
        assistant = [o for o in result.outputs
                     if o.get("data", {}).get("role") == "assistant"]
        self.assertEqual(assistant[0]["finish_reason"], "stop")

    async def test_the_two_are_distinguishable(self):
        """The whole point: a cut-off turn must not look like a finished one."""
        truncated, _ = await _run("length")
        finished, _ = await _run("stop")
        pick = lambda r: [o["finish_reason"] for o in r.outputs
                          if o.get("data", {}).get("role") == "assistant"]
        self.assertNotEqual(pick(truncated), pick(finished))

    async def test_absent_finish_reason_is_none_not_a_false_zero(self):
        """A provider that reports nothing must read as unknown, not as 'stop'."""
        result, _ = await _run(None)
        assistant = [o for o in result.outputs
                     if o.get("data", {}).get("role") == "assistant"]
        self.assertIsNone(assistant[0]["finish_reason"])

    async def test_flag_never_reaches_the_model(self):
        _, completion = await _run("length")
        for request in completion.requests:
            for message in request:
                self.assertNotIn("finish_reason", message)


if __name__ == "__main__":
    unittest.main()
