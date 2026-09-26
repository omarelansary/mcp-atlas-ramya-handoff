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


class MaxTurnsExhaustedError(DynamicMcpEvalError):
    """Turn cap reached, carrying the trajectory built up to that point.

    Observability only. The run is still a failure, still surfaces as HTTP 500
    with ``failure_code`` ``max_turns_exhausted``, and still scores 0 as a
    policy failure. Nothing about the stopping semantics changes: the loop ends
    at exactly the same place, having done exactly the same work.

    Why this exists (thesis defect P1029-D1): this path used to raise a bare
    ``DynamicMcpEvalError`` and the accumulated ``outputs`` were discarded with
    it. P1-029 recorded 8 turn-capped runs as an empty payload -- no messages,
    no cycles, no tool calls -- so the one question those runs exist to answer,
    *why did 50 turns produce nothing*, became permanently unanswerable. Five of
    the eight were a single task, which is exactly the case worth reading.

    The trajectory is the same object a successful run returns, so this adds no
    content category the route did not already emit.
    """

    def __init__(self, message: str, *, outputs: Sequence[dict[str, Any]] = (),
                 cycles: Sequence["DynamicCycleTrace"] = (),
                 usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.outputs = tuple(copy.deepcopy(list(outputs)))
        self.cycles = tuple(cycles)
        self.usage = dict(usage) if usage else {}


@dataclass(frozen=True)
class DynamicSelection:
    """Selector output plus evaluator-safe retention provenance."""

    active_tool_names: tuple[str, ...]
    retained_tool_names: tuple[str, ...] = ()


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
    ) -> Sequence[str] | DynamicSelection:
        """Return ordered active names and optional retention provenance."""


CompletionFn = Callable[..., Awaitable[Any]]


_FORBIDDEN_SOURCE_TOOL_FIELDS = frozenset(
    {
        "enabled_tools",
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
    raw_tool_count: int
    active_tool_names: tuple[str, ...]
    retained_tool_names: tuple[str, ...]
    selector_id: str | None
    provider_tool_count: int
    provider_tools_hash: str
    provider_schema_utf8_bytes: int
    # Which document shape actually reached the provider. Recorded rather than
    # inferred: the shape changes what the model reads, so a record that does
    # not name it cannot say what was measured. Defaulted so every existing
    # construction site and every stored record stays valid.
    tool_document_shape: str = "raw"


@dataclass(frozen=True)
class DynamicMcpEvalResult:
    """In-memory output for deterministic tests and later adapter integration."""

    outputs: tuple[dict[str, Any], ...]
    cycles: tuple[DynamicCycleTrace, ...]
    final_text: str | None
    # Sibling of outputs, not an entry in it: outputs is a message stream and
    # consumers index into it. Token accounting is run metadata.
    usage: dict[str, Any] | None = None


async def run_dynamic_mcp_eval(
    *,
    mcp_client: RawMcpClient,
    selector: DynamicToolSelector,
    completion: CompletionFn,
    model: str,
    messages: Sequence[Message],
    max_turns: int,
    extra_body: dict[str, Any] | None = None,
    tool_document_shape: str = "raw",
) -> DynamicMcpEvalResult:
    """Run a host-selected discovery/model/call loop without changing default eval.

    ``completion`` is injected so P1-016-T0 can use a fake provider. It has the
    same keyword surface as ``create_completion``: model, messages, tools, and
    optional extra_body.

    ``tool_document_shape`` selects how much source metadata reaches the model.
    It defaults to ``raw``, which is byte-for-byte what every run before
    2026-09-15 sent, so nothing already measured is disturbed.
    """

    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if tool_document_shape not in TOOL_DOCUMENT_SHAPES:
        raise DynamicMcpEvalError(
            f"unknown tool document shape {tool_document_shape!r}; "
            f"expected one of {TOOL_DOCUMENT_SHAPES}"
        )

    visible_messages = list(copy.deepcopy(messages))
    outputs: list[dict[str, Any]] = []
    cycles: list[DynamicCycleTrace] = []
    final_text: str | None = None
    # Token accounting for the dynamic route. The fixed-list loop already
    # reports this; without the same here a dynamic run's cost is unmeasurable,
    # which matters most on this route because full exposure re-sends all 126
    # tool schemas every cycle.
    run_usage: dict[str, Any] = {"cycles": 0, "reported_cycles": 0}

    for cycle_index in range(max_turns):
        raw_tools = await mcp_client.list_raw_tools()
        _validate_raw_tools(raw_tools)
        selection = _normalize_selection(
            selector.select(
                raw_tools=copy.deepcopy(raw_tools),
                visible_messages=tuple(copy.deepcopy(visible_messages)),
                cycle_index=cycle_index,
            ),
            raw_tools=raw_tools,
            visible_messages=visible_messages,
        )
        active_names = selection.active_tool_names
        active_name_set = set(active_names)
        active_tools = [tool for tool in raw_tools if tool["name"] in active_name_set]
        provider_tools = raw_tools_to_provider_tools(
            active_tools, document_shape=tool_document_shape
        )
        provider_tool_payload = [tool.model_dump() for tool in provider_tools]
        cycles.append(
            DynamicCycleTrace(
                cycle_index=cycle_index,
                raw_tool_names=tuple(tool["name"] for tool in raw_tools),
                raw_tool_hash=_canonical_hash(raw_tools),
                raw_tool_count=len(raw_tools),
                active_tool_names=active_names,
                retained_tool_names=selection.retained_tool_names,
                selector_id=_selector_id(selector),
                provider_tool_count=len(provider_tools),
                provider_tools_hash=_canonical_hash(provider_tool_payload),
                provider_schema_utf8_bytes=_canonical_utf8_bytes(provider_tool_payload),
                tool_document_shape=tool_document_shape,
            )
        )

        completion_result = await completion(
            model=model,
            messages=visible_messages,
            tools=provider_tools,
            extra_body=extra_body,
        )
        assistant_message = completion_result.message
        run_usage["cycles"] += 1
        # getattr, not attribute access: the completion callable is injectable
        # and test fakes implement only .message. Token accounting must never be
        # the reason a run fails.
        reported = getattr(completion_result, "usage", None)
        if reported:
            run_usage["reported_cycles"] += 1
            for key, value in reported.items():
                if isinstance(value, int):
                    run_usage[key] = run_usage.get(key, 0) + value
        visible_messages.append(assistant_message)
        # `finish_reason` is a sibling of `data`, never inside it, for the same
        # reason as `tool_call_failed` below: `data` is what reaches the provider
        # in `visible_messages`, so a field added there changes what the MODEL
        # reads. It is also not available on the message at all -- litellm keeps
        # it on `choices[0]` -- so nothing downstream could recover it by looking
        # harder at `original_message`. A turn truncated at the token limit is
        # otherwise indistinguishable from one that finished.
        outputs.append({
            "type": "message",
            "data": assistant_message.model_dump(),
            "finish_reason": getattr(completion_result, "finish_reason", None),
        })

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
            # `tool_call_failed` and `tool_name` are siblings of `data`, never
            # inside it. `data` is `ToolCallOutputMessage.model_dump()`, which
            # is also what goes to the provider in `visible_messages` -- adding
            # a field there would change what the MODEL sees, which is a
            # behavioural change, not observability. Recorded out here instead,
            # so the run artefact gains the flag and the agent sees nothing new.
            #
            # Why this exists: `CallToolResponse.is_error` (schema.py:176, the
            # MCP protocol's own error flag) arrived on every tool response and
            # was discarded on this line. A failed call persisted identically
            # to a successful one, so nothing structured could say how often a
            # tool call failed. Measured after the fact by string-matching the
            # error text that happened to survive in `content`: 2,336 of 10,915
            # calls failed in the 2026-08-29 grid (21.4%), and the rate tracked
            # the conditions inversely -- 12.4% under `full_exposure`, 31.6%
            # under `one_shot_b8`. That is not a detail worth losing.
            outputs.append({"type": "message", "data": tool_message.model_dump(),
                            "tool_name": tool_name,
                            "tool_call_failed": bool(response.is_error)})
    else:
        # Message text unchanged: `_dynamic_failure_code` matches on it and the
        # `max_turns_exhausted` code is part of the recorded contract.
        raise MaxTurnsExhaustedError(
            "model did not finish within max_turns",
            outputs=outputs, cycles=cycles, usage=run_usage)

    run_usage["transient_tool_retries"] = getattr(
        mcp_client, "transient_retries", 0
    )

    return DynamicMcpEvalResult(
        outputs=tuple(copy.deepcopy(outputs)),
        cycles=tuple(cycles),
        final_text=final_text,
        usage=run_usage,
    )


#: How a tool's document is built for the provider.
#:
#: ``raw``
#:     name + description + inputSchema. What every run before 2026-09-15 sent,
#:     and the default, so no existing experiment changes.
#: ``with_output``
#:     the same, plus the source's ``outputSchema`` appended to the description
#:     for the tools that carry one. MCP servers publish an output schema and
#:     this route dropped it on the floor: 42 of the 126 MCP-Atlas tools have
#:     one, so a third of the corpus was exposed without the model ever being
#:     told what comes back.
TOOL_DOCUMENT_SHAPES = ("raw", "with_output")

_RETURNS_HEADING = "Returns:"


def _describe_output_schema(tool: dict[str, Any]) -> str | None:
    """The source's own output schema, rendered for a description field.

    Returns ``None`` when the source publishes no output schema, so that the
    description is left byte-identical rather than gaining an empty section.
    """

    output_schema = tool.get("outputSchema")
    if not isinstance(output_schema, dict) or not output_schema:
        return None
    return f"{_RETURNS_HEADING}\n" + json.dumps(
        output_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def raw_tools_to_provider_tools(
    raw_tools: Sequence[dict[str, Any]],
    *,
    document_shape: str = "raw",
) -> list[ToolCallSchema]:
    """Map only provider-supported source fields without inventing metadata.

    ``document_shape`` selects how much of the source's own metadata reaches the
    model. It never invents anything: ``with_output`` copies the server's
    published ``outputSchema`` and nothing else, and a tool without one is
    emitted byte-identically to ``raw``.
    """

    if document_shape not in TOOL_DOCUMENT_SHAPES:
        raise DynamicMcpEvalError(
            f"unknown tool document shape {document_shape!r}; "
            f"expected one of {TOOL_DOCUMENT_SHAPES}"
        )
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
        if document_shape == "with_output":
            returns = _describe_output_schema(tool)
            if returns is not None:
                existing = function.get("description") or ""
                # 4 of the 126 tools carry no description at all; those get the
                # returns block alone rather than a leading blank line.
                function["description"] = (
                    f"{existing}\n\n{returns}" if existing.strip() else returns
                )
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
    forbidden = {
        field_name
        for field_name in tool
        if isinstance(field_name, str)
        and field_name.lower() in _FORBIDDEN_SOURCE_TOOL_FIELDS
    }
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


def _normalize_selection(
    selection: Sequence[str] | DynamicSelection,
    *,
    raw_tools: Sequence[dict[str, Any]],
    visible_messages: Sequence[Message],
) -> DynamicSelection:
    if isinstance(selection, DynamicSelection):
        active_names = _validate_active_names(selection.active_tool_names, raw_tools)
        retained_names = _validate_active_names(selection.retained_tool_names, raw_tools)
    else:
        active_names = _validate_active_names(selection, raw_tools)
        retained_names = ()

    if not set(retained_names).issubset(active_names):
        raise DynamicMcpEvalError("retained tools must be present in the active set")
    called_names = _visible_called_tool_names(visible_messages)
    if any(name not in called_names for name in retained_names):
        raise DynamicMcpEvalError("selector retained a tool that was not previously called")
    return DynamicSelection(
        active_tool_names=active_names,
        retained_tool_names=retained_names,
    )


def _visible_called_tool_names(visible_messages: Sequence[Message]) -> set[str]:
    names: set[str] = set()
    for message in visible_messages:
        if getattr(message, "role", None) != "assistant":
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
    return names


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
    serialized = _canonical_json(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_utf8_bytes(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _selector_id(selector: DynamicToolSelector) -> str | None:
    selector_id = getattr(selector, "selector_id", None)
    return selector_id if isinstance(selector_id, str) and selector_id else None
