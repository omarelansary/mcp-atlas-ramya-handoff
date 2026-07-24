"""Endpoint-level P1-016 tests with fake MCP and completion boundaries."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from mcp_completion.errors import MCPClientToolTimeoutError
from mcp_completion.main import app
from mcp_completion.schema import CallToolResponse, TextContent


@dataclass
class _ToolCall:
    id: str
    function: dict[str, str]


@dataclass
class _AssistantMessage:
    tool_calls: list[_ToolCall] | None = None
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
class _CompletionResult:
    message: _AssistantMessage


class _FakeSandboxMcpClient:
    instances: list["_FakeSandboxMcpClient"] = []
    raw_tools = [
        {
            "name": "files_read",
            "description": "Read a file.",
            "inputSchema": {"type": "object", "properties": {}},
            "server": "filesystem",
        },
        {
            "name": "code_execute",
            "description": "Execute code.",
            "inputSchema": {"type": "object", "properties": {}},
            "server": "mcp-code-executor",
        },
    ]

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.__class__.instances.append(self)

    async def list_raw_tools(self):
        return self.raw_tools

    async def call_tool(self, tool_name, args):
        self.calls.append((tool_name, args))
        return CallToolResponse(
            content=[TextContent(type="text", text="ok")], isError=False
        )


class _TimedOutSandboxMcpClient(_FakeSandboxMcpClient):
    async def call_tool(self, tool_name, args):
        self.calls.append((tool_name, args))
        raise MCPClientToolTimeoutError("tool call timed out")


class _SequenceCompletion:
    def __init__(self, responses):
        self.responses = list(responses)
        self.request_tool_names: list[list[str]] = []

    async def __call__(self, *, model, messages, tools, extra_body):
        self.request_tool_names.append([tool.function["name"] for tool in tools])
        return self.responses.pop(0)


class DynamicEndpointTests(unittest.TestCase):
    def setUp(self):
        _FakeSandboxMcpClient.instances = []

    def test_selected_dynamic_endpoint_forwards_visible_call(self):
        completion = _SequenceCompletion(
            [
                _CompletionResult(
                    _AssistantMessage(
                        tool_calls=[
                            _ToolCall(
                                id="call-1",
                                function={"name": "files_read", "arguments": "{}"},
                            )
                        ]
                    )
                ),
                _CompletionResult(_AssistantMessage(content="done")),
            ]
        )

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Read a file."}],
                    "activeToolNames": ["files_read"],
                    "maxTurns": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(completion.request_tool_names, [["files_read"], ["files_read"]])
        self.assertEqual(_FakeSandboxMcpClient.instances[0].calls, [("files_read", {})])
        self.assertEqual(len(response.json()["outputs"]), 3)
        trace = response.json()["dynamic_trace"]
        self.assertEqual(trace["cycles"][0]["active_tool_names"], ["files_read"])
        self.assertIn("raw_tool_hash", trace["cycles"][0])
        self.assertNotIn("raw_tools", trace["cycles"][0])

    def test_full_dynamic_endpoint_preserves_source_order(self):
        completion = _SequenceCompletion([_CompletionResult(_AssistantMessage(content="done"))])

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Use a tool."}],
                    "maxTurns": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(completion.request_tool_names, [["files_read", "code_execute"]])

    def test_registered_semantic_endpoint_uses_safe_selector_trace(self):
        completion = _SequenceCompletion([_CompletionResult(_AssistantMessage(content="done"))])

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Read a file."}],
                    "selectorId": "dynamic_stateless_semantic",
                    "toolBudget": 1,
                    "maxTurns": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(completion.request_tool_names, [["files_read"]])
        trace = response.json()["dynamic_trace"]["cycles"][0]
        self.assertEqual(trace["selector_id"], "dynamic_stateless_semantic")
        self.assertEqual(trace["raw_tool_count"], 2)
        self.assertEqual(trace["provider_tool_count"], 1)
        self.assertGreater(trace["provider_schema_utf8_bytes"], 0)
        serialized_trace = str(trace)
        for forbidden in ("inputSchema", "description", "arguments", "content"):
            self.assertNotIn(forbidden, serialized_trace)

    def test_registered_full_endpoint_preserves_raw_source_order(self):
        completion = _SequenceCompletion([_CompletionResult(_AssistantMessage(content="done"))])

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Use a tool."}],
                    "selectorId": "dynamic_full",
                    "toolBudget": 1,
                    "maxTurns": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(completion.request_tool_names, [["files_read", "code_execute"]])
        self.assertEqual(
            response.json()["dynamic_trace"]["cycles"][0]["selector_id"],
            "dynamic_full",
        )

    def test_registered_selector_rejects_explicit_active_set_and_evaluator_field(self):
        mixed_mode = TestClient(app).post(
            "/v2/mcp_eval/run_agent_dynamic",
            json={
                "model": "fake/model",
                "messages": [{"role": "user", "content": "Read a file."}],
                "selectorId": "dynamic_stateless_semantic",
                "toolBudget": 1,
                "activeToolNames": ["files_read"],
                "maxTurns": 1,
            },
        )
        evaluator_field = TestClient(app).post(
            "/v2/mcp_eval/run_agent_dynamic",
            json={
                "model": "fake/model",
                "messages": [{"role": "user", "content": "Read a file."}],
                "selectorId": "dynamic_stateless_semantic",
                "toolBudget": 1,
                "GTFA_CLAIMS": "forbidden",
                "maxTurns": 1,
            },
        )
        missing_budget = TestClient(app).post(
            "/v2/mcp_eval/run_agent_dynamic",
            json={
                "model": "fake/model",
                "messages": [{"role": "user", "content": "Read a file."}],
                "selectorId": "dynamic_stateless_semantic",
                "maxTurns": 1,
            },
        )
        unknown_selector = TestClient(app).post(
            "/v2/mcp_eval/run_agent_dynamic",
            json={
                "model": "fake/model",
                "messages": [{"role": "user", "content": "Read a file."}],
                "selectorId": "unregistered",
                "toolBudget": 1,
                "maxTurns": 1,
            },
        )

        self.assertEqual(mixed_mode.status_code, 422)
        self.assertEqual(evaluator_field.status_code, 422)
        self.assertEqual(missing_budget.status_code, 422)
        self.assertEqual(unknown_selector.status_code, 422)
        self.assertEqual(_FakeSandboxMcpClient.instances, [])

    def test_dynamic_endpoint_blocks_hidden_call(self):
        completion = _SequenceCompletion(
            [
                _CompletionResult(
                    _AssistantMessage(
                        tool_calls=[
                            _ToolCall(
                                id="call-hidden",
                                function={"name": "code_execute", "arguments": "{}"},
                            )
                        ]
                    )
                )
            ]
        )

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Read a file."}],
                    "activeToolNames": ["files_read"],
                    "maxTurns": 1,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(_FakeSandboxMcpClient.instances[0].calls, [])
        self.assertEqual(
            response.json(), {"detail": {"failure_code": "hidden_tool_request"}}
        )

    def test_dynamic_endpoint_codes_max_turn_exhaustion_without_error_text(self):
        completion = _SequenceCompletion(
            [
                _CompletionResult(
                    _AssistantMessage(
                        tool_calls=[
                            _ToolCall(
                                id="call-1",
                                function={"name": "files_read", "arguments": "{}"},
                            )
                        ]
                    )
                )
            ]
        )

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _FakeSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Read a file."}],
                    "activeToolNames": ["files_read"],
                    "maxTurns": 1,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": {"failure_code": "max_turns_exhausted"}}
        )

    def test_dynamic_endpoint_codes_mcp_tool_timeout_without_error_text(self):
        completion = _SequenceCompletion(
            [
                _CompletionResult(
                    _AssistantMessage(
                        tool_calls=[
                            _ToolCall(
                                id="call-1",
                                function={"name": "files_read", "arguments": "{}"},
                            )
                        ]
                    )
                )
            ]
        )

        with patch(
            "mcp_completion.agent_eval.SandboxMCPClient", _TimedOutSandboxMcpClient
        ), patch("mcp_completion.agent_eval.create_completion", completion):
            response = TestClient(app).post(
                "/v2/mcp_eval/run_agent_dynamic",
                json={
                    "model": "fake/model",
                    "messages": [{"role": "user", "content": "Read a file."}],
                    "activeToolNames": ["files_read"],
                    "maxTurns": 2,
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": {"failure_code": "mcp_tool_call_timeout"}}
        )


if __name__ == "__main__":
    unittest.main()
