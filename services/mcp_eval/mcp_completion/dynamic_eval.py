"""Deterministic dynamic MCP-Atlas evaluation loop for P1-016.

This is deliberately separate from the official fixed-list ``run_mcp_eval``
path. The host refreshes raw MCP-backed discovery before each completion cycle,
selects the current visible subset, and validates calls before forwarding them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence

from .schema import CallToolResponse, Message, ToolCallOutputMessage, ToolCallSchema


class DynamicMcpEvalError(RuntimeError):
    """Raised when the dynamic evaluation-loop contract is violated."""


class HiddenToolRequestError(DynamicMcpEvalError):
    """Raised when a completion requests a tool outside the active set."""


class RawMcpClient(Protocol):
    """The raw source-object discovery and invocation boundary."""

    async def list_raw_tools(self) -> list[dict[str, Any]]:
        """Return the agent-environment's unprojected MCP tool objects."""

    async def call_tool(self, tool_name: str, args: Any) -> CallToolResponse:
        """Forward one allowed call to the agent environment."""


class DynamicToolSelector(Protocol):
    """Host-side selector; it receives no evaluator or trajectory fields."""

    def select(
        self,
        *,
        raw_tools: Sequence[dict[str, Any]],
        visible_messages: Sequence[Message],
        cycle_index: int,
    ) -> Sequence[str]:
        """Return the ordered active names for this completion cycle."""


CompletionFn = Callable[..., Awaitable[Any]]


_FORBIDDEN_SOURCE_TOOL_FIELDS = frozenset(
    {
        "evaluator",
        "evaluator_labels",
        "gold",
        "gold_labels",
        "gtfa_claims",
        "reference_answer",
        "score",
        "scores",
        "trajectory",
        "trajectories",
        "verifier",
    }
)


@dataclass(frozen=True)
class DynamicCycleTrace:
    """Safe-to-inspect structural record for one dynamic completion cycle."""

    cycle_index: int
    raw_tool_names: tuple[str, ...]
    raw_tool_hash: str
    active_tool_names: tuple[str, ...]
    provider_tools_hash: str


@dataclass(frozen=True)
class DynamicMcpEvalResult:
    """In-memory output for deterministic tests and later adapter integration."""

    outputs: tuple[dict[str, Any], ...]
    cycles: tuple[DynamicCycleTrace, ...]
    final_text: str | None


async def run_dynamic_mcp_eval(
    *,
    mcp_client: RawMcpClient,
    selector: DynamicToolSelector,
    completion: CompletionFn,
    model: str,
    messages: Sequence[Message],
    max_turns: int,
    extra_body: dict[str, Any] | None = None,
) -> DynamicMcpEvalResult:
    """Run a host-selected discovery/model/call loop without changing default eval.

    ``completion`` is injected so P1-016-T0 can use a fake provider. It has the
    same keyword surface as ``create_completion``: model, messages, tools, and
    optional extra_body.
    """

    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    visible_messages = list(copy.deepcopy(messages))
    outputs: list[dict[str, Any]] = []
    cycles: list[DynamicCycleTrace] = []
    final_text: str | None = None

    for cycle_index in range(max_turns):
        raw_tools = await mcp_client.list_raw_tools()
        _validate_raw_tools(raw_tools)
        active_names = _validate_active_names(
            selector.select(
                raw_tools=copy.deepcopy(raw_tools),
                visible_messages=tuple(copy.deepcopy(visible_messages)),
                cycle_index=cycle_index,
            ),
            raw_tools,
        )
        active_name_set = set(active_names)
        active_tools = [tool for tool in raw_tools if tool["name"] in active_name_set]
        provider_tools = raw_tools_to_provider_tools(active_tools)
        cycles.append(
            DynamicCycleTrace(
                cycle_index=cycle_index,
                raw_tool_names=tuple(tool["name"] for tool in raw_tools),
                raw_tool_hash=_canonical_hash(raw_tools),
                active_tool_names=active_names,
                provider_tools_hash=_canonical_hash(
                    [tool.model_dump() for tool in provider_tools]
                ),
            )
        )

        completion_result = await completion(
            model=model,
            messages=visible_messages,
            tools=provider_tools,
            extra_body=extra_body,
        )
        assistant_message = completion_result.message
        visible_messages.append(assistant_message)
        outputs.append({"type": "message", "data": assistant_message.model_dump()})

        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            final_text = assistant_message.content
            break

        call_ids: set[str] = set()
        for tool_call in tool_calls:
            if tool_call.id in call_ids:
                raise DynamicMcpEvalError(f"duplicate tool call id {tool_call.id!r}")
            call_ids.add(tool_call.id)
            tool_name = tool_call.function["name"]
            if tool_name not in active_name_set:
                raise HiddenToolRequestError(
                    f"model requested hidden tool {tool_name!r}; active tools are {list(active_names)!r}"
                )

            arguments = _parse_arguments(tool_call.function["arguments"], tool_name)
            response = await mcp_client.call_tool(tool_name, arguments)
            tool_message = ToolCallOutputMessage(
                role="tool",
                content=response.content,
                tool_call_id=tool_call.id,
            )
            visible_messages.append(tool_message)
            outputs.append({"type": "message", "data": tool_message.model_dump()})
    else:
        raise DynamicMcpEvalError("model did not finish within max_turns")

    return DynamicMcpEvalResult(
        outputs=tuple(copy.deepcopy(outputs)),
        cycles=tuple(cycles),
        final_text=final_text,
    )


def raw_tools_to_provider_tools(raw_tools: Sequence[dict[str, Any]]) -> list[ToolCallSchema]:
    """Map only provider-supported source fields without inventing metadata."""

    provider_tools: list[ToolCallSchema] = []
    for tool in raw_tools:
        _validate_raw_tool(tool)
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": copy.deepcopy(tool["inputSchema"]),
            "strict": False,
        }
        if "description" in tool:
            function["description"] = tool["description"]
        provider_tools.append(ToolCallSchema(type="function", function=function))
    return provider_tools


def _validate_raw_tools(raw_tools: Any) -> None:
    if not isinstance(raw_tools, list):
        raise DynamicMcpEvalError("raw tool discovery must be a list")
    names: set[str] = set()
    for tool in raw_tools:
        _validate_raw_tool(tool)
        name = tool["name"]
        if name in names:
            raise DynamicMcpEvalError(f"duplicate raw tool name {name!r}")
        names.add(name)


def _validate_raw_tool(tool: Any) -> None:
    if not isinstance(tool, dict):
        raise DynamicMcpEvalError("raw tool discovery entries must be objects")
    forbidden = _FORBIDDEN_SOURCE_TOOL_FIELDS.intersection(tool)
    if forbidden:
        raise DynamicMcpEvalError(
            f"evaluator-only fields reached raw discovery: {sorted(forbidden)!r}"
        )
    if not isinstance(tool.get("name"), str) or not tool["name"]:
        raise DynamicMcpEvalError("raw tool name must be a nonempty string")
    if "description" in tool and not isinstance(tool["description"], str):
        raise DynamicMcpEvalError("raw tool description must be a string when present")
    if not isinstance(tool.get("inputSchema"), dict):
        raise DynamicMcpEvalError("raw tool inputSchema must be an object")


def _validate_active_names(
    selected_names: Sequence[str], raw_tools: Sequence[dict[str, Any]]
) -> tuple[str, ...]:
    if not isinstance(selected_names, Sequence) or isinstance(selected_names, (str, bytes)):
        raise DynamicMcpEvalError("selector output must be a sequence of names")
    source_names = {tool["name"] for tool in raw_tools}
    ordered: list[str] = []
    seen: set[str] = set()
    for name in selected_names:
        if not isinstance(name, str) or not name:
            raise DynamicMcpEvalError("selector output contains an invalid tool name")
        if name not in source_names:
            raise DynamicMcpEvalError(f"selector chose undiscovered tool {name!r}")
        if name in seen:
            raise DynamicMcpEvalError(f"selector chose duplicate tool {name!r}")
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _parse_arguments(raw_arguments: Any, tool_name: str) -> dict[str, Any]:
    if not isinstance(raw_arguments, str):
        raise DynamicMcpEvalError(f"tool {tool_name!r} arguments must be a JSON string")
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise DynamicMcpEvalError(
            f"tool {tool_name!r} arguments are not valid JSON"
        ) from error
    if not isinstance(arguments, dict):
        raise DynamicMcpEvalError(f"tool {tool_name!r} arguments must decode to an object")
    return arguments


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
