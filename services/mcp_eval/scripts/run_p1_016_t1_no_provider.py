"""Run P1-016-T1 against a local MCP-Atlas agent environment.

The script has no completion-provider dependency. It uses scripted responses to
exercise the P1-016 dynamic host loop against real `/list-tools` and
`/call-tool` endpoints, and writes a sanitized structural summary only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_completion.dynamic_eval import HiddenToolRequestError, run_dynamic_mcp_eval
from mcp_completion.mcp_client.sandbox_client import SandboxMCPClient
from mcp_completion.schema import CallToolResponse, UserMessage


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


class _ScriptedCompletion:
    def __init__(self, responses: list[_CompletionResult]):
        self._responses = responses
        self.request_tool_names: list[list[str]] = []

    async def __call__(self, *, model, messages, tools, extra_body):
        self.request_tool_names.append([tool.function["name"] for tool in tools])
        if not self._responses:
            raise RuntimeError("scripted completion has no response remaining")
        return self._responses.pop(0)


class _SingleToolSelector:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def select(self, *, raw_tools, visible_messages, cycle_index):
        return [self.tool_name]


class _RecordingRawMcpClient:
    def __init__(self, source: SandboxMCPClient):
        self.source = source
        self.calls: list[dict[str, Any]] = []

    async def list_raw_tools(self) -> list[dict[str, Any]]:
        return await self.source.list_raw_tools()

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> CallToolResponse:
        response = await self.source.call_tool(tool_name, args)
        self.calls.append({"tool_name": tool_name, "is_error": response.is_error})
        return response


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = SandboxMCPClient(args.sandbox_url)
    raw_tools = await source.list_raw_tools()
    raw_names = [tool.get("name") for tool in raw_tools]
    if args.allowed_tool not in raw_names:
        raise RuntimeError(
            f"required allowed tool {args.allowed_tool!r} is not available; discovered {raw_names!r}"
        )

    allowed_client = _RecordingRawMcpClient(source)
    allowed_completion = _ScriptedCompletion(
        [
            _CompletionResult(
                _AssistantMessage(
                    tool_calls=[
                        _ToolCall(
                            id="p1-016-allowed",
                            function={"name": args.allowed_tool, "arguments": "{}"},
                        )
                    ]
                )
            ),
            _CompletionResult(_AssistantMessage(content="transport check complete")),
        ]
    )
    allowed_result = await run_dynamic_mcp_eval(
        mcp_client=allowed_client,
        selector=_SingleToolSelector(args.allowed_tool),
        completion=allowed_completion,
        model="p1-016-scripted-no-provider",
        messages=[
            UserMessage(
                role="user",
                content="Run the predeclared local transport check.",
            )
        ],
        max_turns=2,
    )

    hidden_client = _RecordingRawMcpClient(source)
    hidden_completion = _ScriptedCompletion(
        [
            _CompletionResult(
                _AssistantMessage(
                    tool_calls=[
                        _ToolCall(
                            id="p1-016-hidden",
                            function={"name": args.hidden_tool, "arguments": "{}"},
                        )
                    ]
                )
            )
        ]
    )
    hidden_blocked = False
    try:
        await run_dynamic_mcp_eval(
            mcp_client=hidden_client,
            selector=_SingleToolSelector(args.allowed_tool),
            completion=hidden_completion,
            model="p1-016-scripted-no-provider",
            messages=[
                UserMessage(
                    role="user",
                    content="Run the predeclared local transport check.",
                )
            ],
            max_turns=1,
        )
    except HiddenToolRequestError:
        hidden_blocked = True

    allowed_call_ok = (
        allowed_client.calls == [{"tool_name": args.allowed_tool, "is_error": False}]
    )
    if not allowed_call_ok or not hidden_blocked or hidden_client.calls:
        raise RuntimeError(
            "P1-016-T1 contract failed: expected one non-error allowed call and no hidden forwarding"
        )

    first_cycle = allowed_result.cycles[0]
    return {
        "p1_record": "P1-016-T1",
        "scope": "local no-provider MCP-Atlas dynamic-loop transport check; not task-preservation evidence",
        "sandbox_url": args.sandbox_url,
        "provider_used": False,
        "evaluator_used": False,
        "raw_tool_count": len(first_cycle.raw_tool_names),
        "raw_tool_hash": first_cycle.raw_tool_hash,
        "active_tool_names": list(first_cycle.active_tool_names),
        "provider_request_tool_names": allowed_completion.request_tool_names,
        "allowed_call": allowed_client.calls[0],
        "hidden_tool_name": args.hidden_tool,
        "hidden_call_blocked_before_forwarding": hidden_blocked,
        "hidden_forwarded_call_count": len(hidden_client.calls),
        "success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:1984")
    parser.add_argument("--allowed-tool", default="list_allowed_directories")
    parser.add_argument("--hidden-tool", default="p1_016_hidden_tool")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
