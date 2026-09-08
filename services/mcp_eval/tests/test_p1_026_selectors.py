"""P1-026 adapter selectors: conversion boundary and request wiring.

Fake ranker throughout, so no model weights are needed. These are the
fake-provider tests the adapter's AGENTS.md requires before any live call.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pytest

from mcp_completion.p1_026_selectors import (
    P1026SelectorError,
    build_registered_p1_026_dynamic_selector,
)
from mcp_completion.schema import RunDynamicAgentAPIRequestBody, UserMessage

RAW_TOOLS = [
    {"name": "fetch_fetch", "description": "fetch a url", "inputSchema": {"type": "object"}},
    {"name": "ddg-search_search", "description": "search the web", "inputSchema": {"type": "object"}},
    {"name": "pubmed_lookup", "description": "look up an article", "inputSchema": {"type": "object"}},
]
PROMPT = "find the closest charging station"


def fake_ranker(order):
    def _rank(tools, query):  # noqa: ARG001
        return order

    return _rank


def user(content=PROMPT):
    """A real UserMessage, so schema validation is genuinely exercised."""

    return UserMessage(role="user", content=content)


class _Fn:
    def __init__(self, name):
        self.name = name


class _Call:
    def __init__(self, name):
        self.function = _Fn(name)


def assistant(content=None, tool_names=()):
    """Assistant message stand-in.

    A plain object rather than the real AssistantMessage, which requires a
    litellm original_message this test has no reason to build. The adapter
    reads messages purely through getattr, so this exercises the same path.
    """

    return SimpleNamespace(
        role="assistant", content=content, tool_calls=[_Call(n) for n in tool_names]
    )


def tool_result(text):
    """Tool-output stand-in whose content is a list, as the real type has."""

    return SimpleNamespace(role="tool", content=[SimpleNamespace(type="text", text=text)])


ALL_NAMES = [t["name"] for t in RAW_TOOLS]


# --- request schema ---------------------------------------------------------


@pytest.mark.parametrize(
    "selector_id",
    ["p1_026_full", "p1_026_one_shot", "p1_026_stateless_repeat", "p1_026_registry"],
)
def test_schema_accepts_every_p1_026_selector_id(selector_id) -> None:
    body = RunDynamicAgentAPIRequestBody(
        model="m", messages=[user()], selectorId=selector_id, toolBudget=2
    )
    assert body.selector_id == selector_id


def test_schema_still_accepts_p1_017_selector_ids() -> None:
    body = RunDynamicAgentAPIRequestBody(
        model="m", messages=[user()], selectorId="dynamic_full", toolBudget=2
    )
    assert body.selector_id == "dynamic_full"


# --- construction -----------------------------------------------------------


def test_unknown_selector_id_is_rejected() -> None:
    with pytest.raises(P1026SelectorError):
        build_registered_p1_026_dynamic_selector(
            selector_id="nope", tool_budget=2, messages=[user()], ranker=fake_ranker([])
        )


def test_ranking_conditions_require_a_ranker() -> None:
    for selector_id in ("p1_026_one_shot", "p1_026_stateless_repeat", "p1_026_registry"):
        with pytest.raises(P1026SelectorError):
            build_registered_p1_026_dynamic_selector(
                selector_id=selector_id, tool_budget=2, messages=[user()], ranker=None
            )


def test_full_needs_no_ranker() -> None:
    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_full", tool_budget=2, messages=[user()], ranker=None
    )
    selection = selector.select(raw_tools=RAW_TOOLS, visible_messages=[user()], cycle_index=0)
    assert selection.active_tool_names == tuple(ALL_NAMES)


def test_missing_user_prompt_is_rejected() -> None:
    with pytest.raises(P1026SelectorError):
        build_registered_p1_026_dynamic_selector(
            selector_id="p1_026_registry", tool_budget=2,
            messages=[SimpleNamespace(role="system", content="sys")], ranker=fake_ranker(ALL_NAMES),
        )


# --- conversion boundary ----------------------------------------------------


def test_selector_respects_budget_and_rank_order() -> None:
    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_stateless_repeat", tool_budget=2,
        messages=[user()], ranker=fake_ranker(ALL_NAMES),
    )
    selection = selector.select(raw_tools=RAW_TOOLS, visible_messages=[user()], cycle_index=0)
    assert selection.active_tool_names == (ALL_NAMES[0], ALL_NAMES[1])


def test_registry_retains_a_called_tool_across_cycles() -> None:
    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_registry", tool_budget=2,
        messages=[user()], ranker=fake_ranker(ALL_NAMES),
    )
    selector.select(raw_tools=RAW_TOOLS, visible_messages=[user()], cycle_index=0)

    selector._core_selector.ranker = fake_ranker(list(reversed(ALL_NAMES)))
    messages = [user(), assistant("calling", tool_names=(ALL_NAMES[1],))]
    selection = selector.select(raw_tools=RAW_TOOLS, visible_messages=messages, cycle_index=1)
    assert ALL_NAMES[1] in selection.active_tool_names
    assert selection.retained_tool_names == (ALL_NAMES[1],)


def test_one_shot_freezes_its_first_selection() -> None:
    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_one_shot", tool_budget=2,
        messages=[user()], ranker=fake_ranker(ALL_NAMES),
    )
    first = selector.select(raw_tools=RAW_TOOLS, visible_messages=[user()], cycle_index=0)
    selector._core_selector.ranker = fake_ranker(list(reversed(ALL_NAMES)))
    later = selector.select(
        raw_tools=RAW_TOOLS,
        visible_messages=[user(), assistant("different intent")],
        cycle_index=1,
    )
    assert later.active_tool_names == first.active_tool_names


def test_tool_result_payloads_never_reach_the_ranker() -> None:
    seen: list[str] = []

    def spy_ranker(tools, query):  # noqa: ARG001
        seen.append(query)
        return ALL_NAMES

    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_registry", tool_budget=2, messages=[user()], ranker=spy_ranker
    )
    messages = [
        user(),
        assistant("I will search", tool_names=("ddg-search_search",)),
        tool_result("SECRET_TOOL_RESULT_PAYLOAD"),
    ]
    selector.select(raw_tools=RAW_TOOLS, visible_messages=messages, cycle_index=1)

    assert seen, "ranker was never called"
    assert all("SECRET_TOOL_RESULT_PAYLOAD" not in q for q in seen)
    assert any("ddg-search_search" in q for q in seen)
    assert any(PROMPT in q for q in seen)


def test_raw_tool_without_a_name_is_rejected() -> None:
    selector = build_registered_p1_026_dynamic_selector(
        selector_id="p1_026_full", tool_budget=2, messages=[user()], ranker=None
    )
    with pytest.raises(P1026SelectorError):
        selector.select(
            raw_tools=[{"description": "no name"}], visible_messages=[user()], cycle_index=0
        )


class RealToolCallShapeTests(unittest.TestCase):
    """Extract tool-call names from the PRODUCTION type, not a stand-in.

    The pre-existing test builds its own ``_Fn`` with a ``.name`` attribute and
    its docstring claims that "the adapter reads messages purely through getattr,
    so this exercises the same path". It does not. ``ToolCall.function`` is a
    ``Dict[str, str]``, and ``getattr`` on a dict returns None rather than
    reading the key.

    That gap disabled the registry for an entire 1,065-run grid: prior tool-call
    names were always empty, so nothing was ever retained and the registry arm
    silently ran the stateless_repeat algorithm. These tests use the real schema
    type so the same substitution cannot pass again.
    """

    def test_names_are_extracted_from_a_real_ToolCall(self):
        from mcp_completion.schema import ToolCall
        from mcp_completion.p1_026_selectors import _visible_messages

        class _Msg:
            role = "assistant"
            content = None
            tool_calls = [ToolCall(id="c1", type="function",
                                   function={"name": "fetch_fetch", "arguments": "{}"})]

        visible = _visible_messages([_Msg()])
        self.assertEqual(visible[0].tool_call_names, ("fetch_fetch",),
                         "a dict-shaped function must yield its name")

    def test_attribute_shaped_function_still_works(self):
        """Provider objects are not guaranteed dict-shaped across versions."""
        from mcp_completion.p1_026_selectors import _visible_messages

        class _Fn:
            name = "files_read"

        class _Call:
            function = _Fn()

        class _Msg:
            role = "assistant"
            content = None
            tool_calls = [_Call()]

        self.assertEqual(_visible_messages([_Msg()])[0].tool_call_names,
                         ("files_read",))

    def test_missing_or_blank_names_are_dropped(self):
        from mcp_completion.schema import ToolCall
        from mcp_completion.p1_026_selectors import _visible_messages

        class _Msg:
            role = "assistant"
            content = None
            tool_calls = [ToolCall(id="c1", type="function", function={"arguments": "{}"}),
                          ToolCall(id="c2", type="function", function={"name": ""}),
                          ToolCall(id="c3", type="function", function={"name": "ok_tool"})]

        self.assertEqual(_visible_messages([_Msg()])[0].tool_call_names, ("ok_tool",))
