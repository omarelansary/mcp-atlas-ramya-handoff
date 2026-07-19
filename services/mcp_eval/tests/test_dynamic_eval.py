"""Deterministic P1-016 contract tests; no provider or live server calls."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from mcp_completion.dynamic_eval import (
    DynamicMcpEvalError,
    HiddenToolRequestError,
    run_dynamic_mcp_eval,
)
from mcp_completion.mcp_client.sandbox_client import SandboxMCPClient
from mcp_completion.schema import CallToolResponse, TextContent, UserMessage


@dataclass
class FakeToolCall:
    id: str
    function: dict[str, str]


@dataclass
class FakeAssistantMessage:
    tool_calls: list[FakeToolCall] | None = None
    content: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": call.function}
                for call in self.tool_calls or []
            ]
            or None,
        }


@dataclass
class FakeCompletionResult:
    message: FakeAssistantMessage


class FakeRawMcpClient:
    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_raw_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> CallToolResponse:
        self.calls.append((tool_name, args))
        return CallToolResponse(
            content=[TextContent(type="text", text=f"result:{tool_name}")],
            isError=False,
        )


class SequenceCompletion:
    def __init__(self, responses: list[FakeCompletionResult]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, *, model, messages, tools, extra_body):
        self.requests.append(
            {
                "model": model,
                "messages": [message.model_dump() for message in messages],
                "tools": [tool.model_dump() for tool in tools],
                "extra_body": extra_body,
            }
        )
        return self.responses.pop(0)


class CycleSelector:
    def __init__(self, names_by_cycle: list[list[str]]):
        self.names_by_cycle = names_by_cycle
        self.calls: list[dict[str, Any]] = []

    def select(self, *, raw_tools, visible_messages, cycle_index):
        self.calls.append(
            {
                "raw_tool_names": [tool["name"] for tool in raw_tools],
                "visible_message_count": len(visible_messages),
                "cycle_index": cycle_index,
            }
        )
        return self.names_by_cycle[cycle_index]


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "files_read",
            "description": "Read a file.",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "server": "filesystem",
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "code_execute",
            "description": "Execute code.",
            "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}},
            "server": "mcp-code-executor",
        },
    ]


class DynamicMcpEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_dynamic_preserves_raw_source_order(self):
        mcp_client = FakeRawMcpClient(_tools())
        completion = SequenceCompletion(
            [FakeCompletionResult(FakeAssistantMessage(content="done"))]
        )

        result = await run_dynamic_mcp_eval(
            mcp_client=mcp_client,
            selector=CycleSelector([["files_read", "code_execute"]]),
            completion=completion,
            model="fake/model",
            messages=[UserMessage(role="user", content="Choose any available tool.")],
            max_turns=1,
        )

        self.assertEqual(result.cycles[0].raw_tool_names, ("files_read", "code_execute"))
        self.assertEqual(result.cycles[0].active_tool_names, ("files_read", "code_execute"))
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.requests[0]["tools"]],
            ["files_read", "code_execute"],
        )

    async def test_refreshes_active_tools_and_forwards_only_visible_call(self):
        mcp_client = FakeRawMcpClient(_tools())
        selector = CycleSelector([["files_read"], ["code_execute"]])
        completion = SequenceCompletion(
            [
                FakeCompletionResult(
                    FakeAssistantMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call-1",
                                function={"name": "files_read", "arguments": '{"path":"/tmp/a"}'},
                            )
                        ]
                    )
                ),
                FakeCompletionResult(FakeAssistantMessage(content="done")),
            ]
        )

        result = await run_dynamic_mcp_eval(
            mcp_client=mcp_client,
            selector=selector,
            completion=completion,
            model="fake/model",
            messages=[UserMessage(role="user", content="Read the file, then calculate.")],
            max_turns=2,
        )

        self.assertEqual(
            [cycle.active_tool_names for cycle in result.cycles],
            [("files_read",), ("code_execute",)],
        )
        self.assertEqual(mcp_client.calls, [("files_read", {"path": "/tmp/a"})])
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.requests[0]["tools"]],
            ["files_read"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in completion.requests[1]["tools"]],
            ["code_execute"],
        )
        self.assertEqual(completion.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(selector.calls[1]["visible_message_count"], 3)
        self.assertEqual(result.final_text, "done")

    async def test_hidden_tool_request_is_blocked_before_forwarding(self):
        mcp_client = FakeRawMcpClient(_tools())
        completion = SequenceCompletion(
            [
                FakeCompletionResult(
                    FakeAssistantMessage(
                        tool_calls=[
                            FakeToolCall(
                                id="call-hidden",
                                function={"name": "code_execute", "arguments": '{"code":"1+1"}'},
                            )
                        ]
                    )
                )
            ]
        )

        with self.assertRaises(HiddenToolRequestError):
            await run_dynamic_mcp_eval(
                mcp_client=mcp_client,
                selector=CycleSelector([["files_read"]]),
                completion=completion,
                model="fake/model",
                messages=[UserMessage(role="user", content="Read a file.")],
                max_turns=1,
            )

        self.assertEqual(mcp_client.calls, [])

    async def test_rejects_evaluator_fields_before_selector_or_model(self):
        contaminated_tools = _tools()
        contaminated_tools[0]["gold_labels"] = ["files_read"]
        selector = CycleSelector([["files_read"]])
        completion = SequenceCompletion([])

        with self.assertRaises(DynamicMcpEvalError):
            await run_dynamic_mcp_eval(
                mcp_client=FakeRawMcpClient(contaminated_tools),
                selector=selector,
                completion=completion,
                model="fake/model",
                messages=[UserMessage(role="user", content="Read a file.")],
                max_turns=1,
            )

        self.assertEqual(selector.calls, [])
        self.assertEqual(completion.requests, [])


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse(self.payload)


class SandboxClientCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_discovery_addition_does_not_change_static_filtering(self):
        payload = _tools()
        with patch(
            "mcp_completion.mcp_client.sandbox_client.httpx.AsyncClient",
            return_value=_FakeHttpClient(payload),
        ):
            client = SandboxMCPClient("http://sandbox", enabled_tools=["files_read"])
            self.assertEqual(
                [tool.name for tool in await client.list_tools()], ["files_read"]
            )
            self.assertEqual(
                [tool["name"] for tool in await client.list_raw_tools()],
                ["files_read", "code_execute"],
            )


if __name__ == "__main__":
    unittest.main()
