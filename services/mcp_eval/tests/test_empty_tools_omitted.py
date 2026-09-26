"""A no-tools condition must not send `tools: []`.

Exposing zero tools is a legitimate active set -- it measures what the model
solves from parametric knowledge alone, which is the floor every tool-exposure
number should be read against. It is not a degenerate case.

But the gateway refuses the empty array. Verified live against ScaDS,
2026-09-16, on zai-org/GLM-5.3-Flash:

    tools OMITTED                -> 200 OK
    "tools": []                  -> 400 "`tools` must not be an empty array"
    "tools": [] + tool_choice    -> 400, same

So an unguarded no-tools arm fails every run identically, and the failure looks
like a model or endpoint fault rather than a request-construction bug.
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from litellm.types.utils import Message as LiteLLMMessage

from mcp_completion.llm import create_completion
from mcp_completion.schema import ToolCallSchema, UserMessage

TOOL = ToolCallSchema(type="function", function={
    "name": "t", "description": "d",
    "parameters": {"type": "object"}, "strict": False})


def _response():
    """A real litellm Message: AssistantMessage validates its type."""
    message = LiteLLMMessage(content="ok", role="assistant", tool_calls=None)
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    return type("R", (), {"choices": [choice], "usage": None})()


class EmptyToolsOmittedTests(unittest.IsolatedAsyncioTestCase):
    async def _kwargs(self, tools):
        with patch("mcp_completion.llm.litellm.acompletion",
                   new=AsyncMock(return_value=_response())) as call:
            await create_completion(
                model="m",
                messages=[UserMessage(role="user", content="hi")],
                tools=tools)
        return call.await_args.kwargs

    async def test_empty_tools_are_omitted_entirely(self):
        kwargs = await self._kwargs([])
        self.assertNotIn("tools", kwargs)

    async def test_non_empty_tools_are_still_sent(self):
        kwargs = await self._kwargs([TOOL])
        self.assertIn("tools", kwargs)
        self.assertEqual(len(kwargs["tools"]), 1)

    async def test_omitting_tools_does_not_disturb_extra_body(self):
        """The no-tools arm still needs its reasoning-effort setting."""
        with patch("mcp_completion.llm.litellm.acompletion",
                   new=AsyncMock(return_value=_response())) as call:
            await create_completion(
                model="m", messages=[UserMessage(role="user", content="hi")],
                tools=[], extra_body={"reasoning_effort": "high"})
        kwargs = call.await_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "high"})


if __name__ == "__main__":
    unittest.main()
