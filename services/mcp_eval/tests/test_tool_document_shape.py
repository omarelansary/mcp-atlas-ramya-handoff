"""The tool document shape changes what the model reads, so it is pinned.

MCP servers publish an ``outputSchema`` and this route dropped it: 42 of the 126
MCP-Atlas tools carry one, so a third of the corpus was exposed without the model
ever being told what comes back. ``with_output`` copies that schema into the
description.

The first test is the important one. ``raw`` must stay byte-for-byte what every
run before 2026-09-15 sent, because the alternative is that adding an option
silently re-specifies every experiment already on record.

Nothing is invented: a tool whose source publishes no output schema is emitted
identically under both shapes, and ``parameters`` is never touched, so the tool
stays callable whichever shape is used.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from mcp_completion.dynamic_eval import (
    DynamicMcpEvalError,
    DynamicSelection,
    raw_tools_to_provider_tools,
    run_dynamic_mcp_eval,
)
from mcp_completion.schema import (
    CallToolResponse,
    RunDynamicAgentAPIRequestBody,
    TextContent,
    UserMessage,
)

WITH_OUTPUT = {
    "name": "calculator_calculate",
    "description": "Calculates the expression.",
    "inputSchema": {"type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]},
    "outputSchema": {"type": "object",
                     "properties": {"result": {"type": "string"}},
                     "required": ["result"]},
}
WITHOUT_OUTPUT = {
    "name": "git_git_log",
    "description": "Shows the commit logs",
    "inputSchema": {"type": "object",
                    "properties": {"repo_path": {"type": "string"}},
                    "required": ["repo_path"]},
    "outputSchema": None,
}
NO_DESCRIPTION = {
    "name": "undocumented_tool",
    "inputSchema": {"type": "object", "properties": {}},
    "outputSchema": {"type": "object",
                     "properties": {"ok": {"type": "boolean"}}},
}


class ToolDocumentShapeTests(unittest.TestCase):
    def test_raw_is_unchanged_by_the_new_option(self):
        """The regression guard: default behaviour must not move."""
        tools = [WITH_OUTPUT, WITHOUT_OUTPUT, NO_DESCRIPTION]
        default = [t.model_dump() for t in raw_tools_to_provider_tools(tools)]
        explicit = [t.model_dump() for t in
                    raw_tools_to_provider_tools(tools, document_shape="raw")]
        self.assertEqual(default, explicit)
        for tool in default:
            self.assertNotIn("Returns:", tool["function"].get("description") or "")

    def test_with_output_appends_the_sources_own_schema(self):
        tool = raw_tools_to_provider_tools(
            [WITH_OUTPUT], document_shape="with_output")[0]
        description = tool.function["description"]
        self.assertTrue(description.startswith("Calculates the expression."))
        self.assertIn("Returns:", description)
        self.assertIn('"result"', description)

    def test_tool_without_an_output_schema_is_untouched(self):
        """Nothing is invented for a source that publishes nothing."""
        raw = raw_tools_to_provider_tools([WITHOUT_OUTPUT])[0].model_dump()
        enriched = raw_tools_to_provider_tools(
            [WITHOUT_OUTPUT], document_shape="with_output")[0].model_dump()
        self.assertEqual(raw, enriched)

    def test_missing_description_gets_the_returns_block_alone(self):
        tool = raw_tools_to_provider_tools(
            [NO_DESCRIPTION], document_shape="with_output")[0]
        self.assertTrue(tool.function["description"].startswith("Returns:"))

    def test_parameters_are_never_altered_so_the_tool_stays_callable(self):
        for shape in ("raw", "with_output"):
            tool = raw_tools_to_provider_tools(
                [WITH_OUTPUT], document_shape=shape)[0]
            self.assertEqual(tool.function["parameters"],
                             WITH_OUTPUT["inputSchema"])

    def test_unknown_shape_is_refused(self):
        with self.assertRaises(DynamicMcpEvalError):
            raw_tools_to_provider_tools([WITH_OUTPUT], document_shape="enriched")

    def test_request_body_defaults_to_raw(self):
        body = RunDynamicAgentAPIRequestBody(
            model="m", messages=[{"role": "user", "content": "hi"}],
            activeToolNames=["calculator_calculate"])
        self.assertEqual(body.tool_document_shape, "raw")

    def test_request_body_accepts_the_alias(self):
        body = RunDynamicAgentAPIRequestBody(
            model="m", messages=[{"role": "user", "content": "hi"}],
            activeToolNames=["calculator_calculate"],
            toolDocumentShape="with_output")
        self.assertEqual(body.tool_document_shape, "with_output")

    def test_request_body_rejects_an_unknown_shape(self):
        with self.assertRaises(Exception):
            RunDynamicAgentAPIRequestBody(
                model="m", messages=[{"role": "user", "content": "hi"}],
                activeToolNames=["calculator_calculate"],
                toolDocumentShape="whatever")


# --- the shape must reach the provider and be recorded ----------------------

@dataclass
class _AssistantMessage:
    content: str | None = None
    tool_calls: list | None = None

    def model_dump(self) -> dict[str, Any]:
        return {"role": "assistant", "content": self.content, "tool_calls": None}


@dataclass
class _CompletionResult:
    message: _AssistantMessage


class _Client:
    async def list_raw_tools(self):
        return [dict(WITH_OUTPUT)]

    async def call_tool(self, tool_name, args):
        return CallToolResponse(
            content=[TextContent(type="text", text="ok")], isError=False)


class _Completion:
    def __init__(self):
        self.tools_seen: list = []

    async def __call__(self, *, model, messages, tools, extra_body):
        self.tools_seen.append([t.model_dump() for t in tools])
        return _CompletionResult(_AssistantMessage(content="done"))


@dataclass
class _Selector:
    selector_id: str = "test"
    calls: list = field(default_factory=list)

    def select(self, *, raw_tools, visible_messages, cycle_index):
        self.calls.append(cycle_index)
        return DynamicSelection(active_tool_names=("calculator_calculate",))


class ShapeReachesTheProviderTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, shape):
        completion = _Completion()
        result = await run_dynamic_mcp_eval(
            mcp_client=_Client(), selector=_Selector(), completion=completion,
            model="fake/model",
            messages=[UserMessage(role="user", content="go")],
            max_turns=3, tool_document_shape=shape)
        return result, completion

    async def test_with_output_reaches_the_model(self):
        result, completion = await self._run("with_output")
        description = completion.tools_seen[0][0]["function"]["description"]
        self.assertIn("Returns:", description)
        self.assertEqual(result.cycles[0].tool_document_shape, "with_output")

    async def test_raw_does_not(self):
        result, completion = await self._run("raw")
        description = completion.tools_seen[0][0]["function"]["description"]
        self.assertNotIn("Returns:", description)
        self.assertEqual(result.cycles[0].tool_document_shape, "raw")

    async def test_schema_bytes_differ_so_cost_is_measurable(self):
        """The two shapes must be distinguishable in the recorded cost figure."""
        raw, _ = await self._run("raw")
        enriched, _ = await self._run("with_output")
        self.assertGreater(enriched.cycles[0].provider_schema_utf8_bytes,
                           raw.cycles[0].provider_schema_utf8_bytes)

    async def test_unknown_shape_is_refused_before_any_call(self):
        with self.assertRaises(DynamicMcpEvalError):
            await self._run("enriched")


if __name__ == "__main__":
    unittest.main()
