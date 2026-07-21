"""MCP-Atlas adapter for the thesis-owned frozen P1-017 selector policy.

Selection logic belongs to ``active_registry_core`` in thesis-main. This module
only converts MCP-Atlas raw tool/message objects to source-neutral contracts
and converts the core selection back to the dynamic host's trace type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from active_registry_core import (
    DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
    DYNAMIC_FULL_SELECTOR_ID,
    DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID,
    REGISTERED_DYNAMIC_SELECTOR_IDS,
    P1017SelectorConfigurationError,
    RegisteredP1017Selector,
    ToolDescriptor,
    VisibleMessage,
    build_registered_p1_017_selector,
)

from .dynamic_eval import DynamicSelection
from .schema import Message


DynamicSelectorConfigurationError = P1017SelectorConfigurationError


@dataclass(frozen=True)
class RegisteredDynamicSelector:
    """Adapt MCP-Atlas discovery and messages to the thesis selector core."""

    selector_id: str
    tool_budget: int
    retention_capacity: int = 2
    _core_selector: RegisteredP1017Selector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            core_selector = RegisteredP1017Selector(
                selector_id=self.selector_id,
                tool_budget=self.tool_budget,
                retention_capacity=self.retention_capacity,
            )
        except P1017SelectorConfigurationError as error:
            raise DynamicSelectorConfigurationError(str(error)) from error
        object.__setattr__(self, "_core_selector", core_selector)

    def select(
        self,
        *,
        raw_tools: Sequence[dict[str, Any]],
        visible_messages: Sequence[Message],
        cycle_index: int,
    ) -> DynamicSelection:
        del cycle_index  # Frozen P1-017 policies have no cycle-number feature.
        selection = self._core_selector.select(
            tools=tuple(_tool_descriptor(raw_tool) for raw_tool in raw_tools),
            visible_messages=tuple(
                _visible_message(message) for message in visible_messages
            ),
        )
        return DynamicSelection(
            active_tool_names=selection.active_tool_names,
            retained_tool_names=selection.retained_tool_names,
        )


def build_registered_dynamic_selector(
    *, selector_id: str, tool_budget: int
) -> RegisteredDynamicSelector:
    """Build the MCP-Atlas adapter for one frozen P1-017 condition."""

    try:
        build_registered_p1_017_selector(
            selector_id=selector_id,
            tool_budget=tool_budget,
        )
    except P1017SelectorConfigurationError as error:
        raise DynamicSelectorConfigurationError(str(error)) from error
    return RegisteredDynamicSelector(selector_id=selector_id, tool_budget=tool_budget)


def _tool_descriptor(raw_tool: dict[str, Any]) -> ToolDescriptor:
    return ToolDescriptor(
        name=raw_tool["name"],
        description=raw_tool.get("description"),
        input_schema=raw_tool.get("inputSchema"),
    )


def _visible_message(message: Message) -> VisibleMessage:
    tool_call_names: list[str] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        name = (
            function.get("name")
            if isinstance(function, dict)
            else getattr(function, "name", None)
        )
        if isinstance(name, str) and name:
            tool_call_names.append(name)
    content = getattr(message, "content", None)
    return VisibleMessage(
        role=getattr(message, "role"),
        content=content if isinstance(content, str) else None,
        tool_call_names=tuple(tool_call_names),
    )
