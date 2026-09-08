"""MCP-Atlas adapter for the thesis-owned P1-026 registered selectors.

Selection logic belongs to ``active_registry_core`` in thesis-main. This module
only converts MCP-Atlas raw tool and message objects into the source-neutral
contracts the core expects, and converts the core's selection back into the
dynamic host's trace type. It mirrors ``dynamic_selectors.py``, which does the
same for P1-017.

The ranker is supplied by the caller rather than constructed here, for two
reasons. It keeps this module free of model dependencies, matching the adapter
boundary rule. And it lets the host build one ranker per process, with the
126 tool-document embeddings computed once at startup, instead of reloading
weights on every cycle of every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from active_registry_core import (
    P1_026_FULL_SELECTOR_ID,
    REGISTERED_P1_026_SELECTOR_IDS,
    P1026SelectorConfigurationError,
    RegisteredP1026Selector,
    ToolDescriptor,
    VisibleMessage,
    build_registered_p1_026_selector,
)

from .dynamic_eval import DynamicSelection
from .schema import Message

P1026SelectorError = P1026SelectorConfigurationError


def _tool_descriptors(raw_tools: Sequence[dict[str, Any]]) -> tuple[ToolDescriptor, ...]:
    """Convert MCP-Atlas raw discovery records to source-neutral descriptors.

    Only the three fields a selector is permitted to see are carried across;
    anything else present on the raw record is dropped here rather than
    filtered downstream.
    """

    descriptors: list[ToolDescriptor] = []
    for raw in raw_tools:
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise P1026SelectorError("raw tool record is missing a usable name")
        descriptors.append(
            ToolDescriptor(
                name=name,
                description=raw.get("description"),
                input_schema=raw.get("inputSchema") or raw.get("input_schema"),
            )
        )
    return tuple(descriptors)


def _visible_messages(messages: Sequence[Message]) -> tuple[VisibleMessage, ...]:
    """Convert host messages to the source-neutral visible-message contract.

    Tool-call *names* are carried; tool-call arguments and tool-result payloads
    are not. The core's query builder additionally ignores ``role == "tool"``
    content, so result text cannot reach the ranker by either path.
    """

    converted: list[VisibleMessage] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        names: list[str] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            # ``ToolCall.function`` is a ``Dict[str, str]`` (schema.py), and
            # ``getattr`` on a dict returns None -- it does not read keys. The
            # previous version used getattr alone, so this list was empty for
            # every message of every run.
            #
            # **That silently disabled the registry.** ``retained`` derives from
            # these names, so it was always empty and the registry branch reduced
            # to ``selected = ranked[:budget]`` -- byte-identical to
            # stateless_repeat. The 2026-08-27 grid recorded 0 non-empty
            # ``retained_tool_names`` across all 1,355 registry_b8 cycles, and
            # H1 unknowingly compared two instances of the same algorithm.
            #
            # Both shapes are handled because the host builds messages from
            # provider objects whose attribute/dict form is not guaranteed
            # across LiteLLM versions.
            if isinstance(function, dict):
                name = function.get("name")
            else:
                name = getattr(function, "name", None) if function else None
            if isinstance(name, str) and name:
                names.append(name)
        content = getattr(message, "content", None)
        converted.append(
            VisibleMessage(
                role=role,
                content=content if isinstance(content, str) else None,
                tool_call_names=tuple(names),
            )
        )
    return tuple(converted)


def _initial_prompt(messages: Sequence[Message]) -> str:
    """The task prompt: the first user message of the run."""

    for message in messages:
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
    raise P1026SelectorError("no user prompt found in the request messages")


@dataclass
class RegisteredP1026DynamicSelector:
    """Adapt MCP-Atlas discovery and messages to the P1-026 selector core."""

    selector_id: str
    tool_budget: int
    prompt: str
    ranker: Any = None
    _core_selector: RegisteredP1026Selector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._core_selector = build_registered_p1_026_selector(
            selector_id=self.selector_id,
            tool_budget=self.tool_budget,
            prompt=self.prompt,
            ranker=self.ranker,
        )

    def select(
        self,
        *,
        raw_tools: Sequence[dict[str, Any]],
        visible_messages: Sequence[Message],
        cycle_index: int,
    ) -> DynamicSelection:
        selection = self._core_selector.select(
            tools=_tool_descriptors(raw_tools),
            visible_messages=_visible_messages(visible_messages),
        )
        return DynamicSelection(
            active_tool_names=tuple(selection.active_tool_names),
            retained_tool_names=tuple(selection.retained_tool_names),
        )


def build_registered_p1_026_dynamic_selector(
    *,
    selector_id: str,
    tool_budget: int,
    messages: Sequence[Message],
    ranker: Any = None,
) -> RegisteredP1026DynamicSelector:
    """Build one P1-026 selector for a single dynamic run."""

    if selector_id not in REGISTERED_P1_026_SELECTOR_IDS:
        raise P1026SelectorError(f"unknown P1-026 selector id {selector_id!r}")
    if ranker is None and selector_id != P1_026_FULL_SELECTOR_ID:
        raise P1026SelectorError(
            f"selector {selector_id!r} requires a ranker supplied by the host"
        )
    return RegisteredP1026DynamicSelector(
        selector_id=selector_id,
        tool_budget=tool_budget,
        prompt=_initial_prompt(messages),
        ranker=ranker,
    )
