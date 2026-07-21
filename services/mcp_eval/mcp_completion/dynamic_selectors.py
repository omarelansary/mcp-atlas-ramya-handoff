"""Registered non-oracle selectors for the P1-017 dynamic MCP route."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .dynamic_eval import DynamicSelection
from .schema import Message


DYNAMIC_FULL_SELECTOR_ID = "dynamic_full"
DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID = "dynamic_stateless_semantic"
DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID = "dynamic_called_tool_retention"

REGISTERED_DYNAMIC_SELECTOR_IDS = frozenset(
    {
        DYNAMIC_FULL_SELECTOR_ID,
        DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID,
        DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DynamicSelectorConfigurationError(ValueError):
    """Raised when a registered selector condition is invalid."""


@dataclass(frozen=True)
class RegisteredDynamicSelector:
    """Select raw MCP tool names using only the P1-017 visible inputs."""

    selector_id: str
    tool_budget: int
    retention_capacity: int = 2

    def __post_init__(self) -> None:
        if self.selector_id not in REGISTERED_DYNAMIC_SELECTOR_IDS:
            raise DynamicSelectorConfigurationError(
                f"unknown dynamic selector id {self.selector_id!r}"
            )
        if not isinstance(self.tool_budget, int) or self.tool_budget < 1:
            raise DynamicSelectorConfigurationError("tool_budget must be a positive integer")
        if not isinstance(self.retention_capacity, int) or self.retention_capacity < 0:
            raise DynamicSelectorConfigurationError(
                "retention_capacity must be a nonnegative integer"
            )

    def select(
        self,
        *,
        raw_tools: Sequence[dict[str, Any]],
        visible_messages: Sequence[Message],
        cycle_index: int,
    ) -> DynamicSelection:
        del cycle_index  # The frozen policies use no cycle-number feature.
        raw_names = tuple(tool["name"] for tool in raw_tools)

        if self.selector_id == DYNAMIC_FULL_SELECTOR_ID:
            return DynamicSelection(active_tool_names=raw_names)

        ranked_names = _rank_semantic_anchor_tools(raw_tools, visible_messages)
        if self.selector_id == DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID:
            return DynamicSelection(
                active_tool_names=tuple(ranked_names[: self.tool_budget])
            )

        retained_names = _recent_called_names(
            visible_messages=visible_messages,
            raw_names=set(raw_names),
            capacity=min(self.retention_capacity, self.tool_budget),
        )
        selected_names = list(retained_names)
        for name in ranked_names:
            if len(selected_names) >= self.tool_budget:
                break
            if name not in selected_names:
                selected_names.append(name)
        return DynamicSelection(
            active_tool_names=tuple(selected_names),
            retained_tool_names=tuple(retained_names),
        )


def build_registered_dynamic_selector(
    *, selector_id: str, tool_budget: int
) -> RegisteredDynamicSelector:
    """Build one frozen P1-017 selector condition."""

    return RegisteredDynamicSelector(selector_id=selector_id, tool_budget=tool_budget)


def _rank_semantic_anchor_tools(
    raw_tools: Sequence[dict[str, Any]], visible_messages: Sequence[Message]
) -> list[str]:
    query_tokens = _visible_request_tokens(visible_messages)
    prefix_evidence: dict[str, int] = {}
    lexical_scores: list[int] = []

    for tool in raw_tools:
        lexical_score = len(query_tokens.intersection(_tokens(_tool_text(tool))))
        lexical_scores.append(lexical_score)
        prefix = _server_prefix(tool["name"])
        prefix_evidence[prefix] = max(prefix_evidence.get(prefix, 0), lexical_score)

    scored: list[tuple[int, int, str]] = []
    for index, tool in enumerate(raw_tools):
        name = tool["name"]
        score = (
            lexical_scores[index] * 100
            + prefix_evidence[_server_prefix(name)] * 10
            + _visible_operation_prior(name)
        )
        scored.append((-score, index, name))
    scored.sort()
    return [name for _, _, name in scored]


def _visible_request_tokens(visible_messages: Sequence[Message]) -> set[str]:
    """Use user/system text only; tool results and assistant text stay excluded."""

    text_parts: list[str] = []
    for message in visible_messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role in {"system", "user"} and isinstance(content, str):
            text_parts.append(content)
    return _tokens(" ".join(text_parts))


def _recent_called_names(
    *, visible_messages: Sequence[Message], raw_names: set[str], capacity: int
) -> list[str]:
    """Retain only distinct, model-requested names that remain discoverable."""

    if capacity == 0:
        return []

    called_names: list[str] = []
    for message in visible_messages:
        if getattr(message, "role", None) != "assistant":
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = _tool_call_name(tool_call)
            if name:
                called_names.append(name)

    retained_names: list[str] = []
    seen: set[str] = set()
    for name in reversed(called_names):
        if name in raw_names and name not in seen:
            retained_names.append(name)
            seen.add(name)
        if len(retained_names) >= capacity:
            break
    return retained_names


def _tool_call_name(tool_call: Any) -> str | None:
    function = (
        tool_call.get("function")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "function", None)
    )
    name = function.get("name") if isinstance(function, dict) else getattr(function, "name", None)
    return name if isinstance(name, str) and name else None


def _tool_text(tool: dict[str, Any]) -> str:
    description = tool.get("description")
    return f"{tool['name']} {description if isinstance(description, str) else ''}"


def _server_prefix(tool_name: str) -> str:
    return tool_name.split("_", 1)[0]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _visible_operation_prior(tool_name: str) -> int:
    """Frozen visible-name anchors reused by the P1-017 semantic condition."""

    name = tool_name.lower()
    priority_patterns: tuple[tuple[int, tuple[str, ...]], ...] = (
        (35, ("execute_code", "run_command")),
        (34, ("calculate",)),
        (32, ("git_log",)),
        (31, ("list_allowed_directories", "show_security_rules", "get_config")),
        (30, ("list_directory",)),
        (28, ("read_file",)),
        (27, ("read_text_file",)),
        (20, ("execute",)),
        (18, ("directory",)),
        (16, ("read",)),
        (8, ("search",)),
        (6, ("get_", "fetch")),
    )
    for score, patterns in priority_patterns:
        if any(pattern in name for pattern in patterns):
            return score
    return 0
