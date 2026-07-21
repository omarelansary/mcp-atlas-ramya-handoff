"""P1-017 registered-selector tests without provider or MCP calls."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from mcp_completion.dynamic_selectors import (
    DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
    DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID,
    DynamicSelectorConfigurationError,
    build_registered_dynamic_selector,
)
from mcp_completion.schema import UserMessage


@dataclass
class _ToolCall:
    function: dict[str, str]


@dataclass
class _AssistantMessage:
    tool_calls: list[_ToolCall]
    role: str = "assistant"


def _tools() -> list[dict]:
    return [
        {
            "name": "filesystem_read_file",
            "description": "Read a text file.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "calendar_create_event",
            "description": "Create a calendar event.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "weather_get_forecast",
            "description": "Get a weather forecast.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


class RegisteredDynamicSelectorTests(unittest.TestCase):
    def test_stateless_semantic_selection_is_deterministic_from_visible_fields(self):
        selector = build_registered_dynamic_selector(
            selector_id=DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID,
            tool_budget=2,
        )
        messages = [UserMessage(role="user", content="Read a file from disk.")]

        first = selector.select(raw_tools=_tools(), visible_messages=messages, cycle_index=0)
        second = selector.select(raw_tools=_tools(), visible_messages=messages, cycle_index=1)

        self.assertEqual(first, second)
        self.assertEqual(
            first.active_tool_names,
            ("filesystem_read_file", "weather_get_forecast"),
        )
        self.assertEqual(first.retained_tool_names, ())

    def test_called_tool_retention_changes_selection_only_after_visible_call(self):
        selector = build_registered_dynamic_selector(
            selector_id=DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
            tool_budget=1,
        )
        user_message = UserMessage(role="user", content="Read a file from disk.")

        initial = selector.select(
            raw_tools=_tools(), visible_messages=[user_message], cycle_index=0
        )
        after_visible_call = selector.select(
            raw_tools=_tools(),
            visible_messages=[
                user_message,
                _AssistantMessage(
                    tool_calls=[_ToolCall(function={"name": "calendar_create_event"})]
                ),
            ],
            cycle_index=1,
        )

        self.assertEqual(initial.active_tool_names, ("filesystem_read_file",))
        self.assertEqual(
            after_visible_call.active_tool_names, ("calendar_create_event",)
        )
        self.assertEqual(
            after_visible_call.retained_tool_names, ("calendar_create_event",)
        )

    def test_retention_capacity_never_promotes_an_uncalled_tool(self):
        selector = build_registered_dynamic_selector(
            selector_id=DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
            tool_budget=2,
        )
        selection = selector.select(
            raw_tools=_tools(),
            visible_messages=[
                UserMessage(role="user", content="Read a file from disk."),
                _AssistantMessage(
                    tool_calls=[
                        _ToolCall(function={"name": "weather_get_forecast"}),
                        _ToolCall(function={"name": "calendar_create_event"}),
                    ]
                ),
            ],
            cycle_index=1,
        )

        self.assertEqual(
            selection.retained_tool_names,
            ("calendar_create_event", "weather_get_forecast"),
        )
        self.assertEqual(selection.active_tool_names, selection.retained_tool_names)
        self.assertNotIn("filesystem_read_file", selection.retained_tool_names)

    def test_retention_never_exceeds_the_selected_budget(self):
        selector = build_registered_dynamic_selector(
            selector_id=DYNAMIC_CALLED_TOOL_RETENTION_SELECTOR_ID,
            tool_budget=1,
        )
        selection = selector.select(
            raw_tools=_tools(),
            visible_messages=[
                UserMessage(role="user", content="Read a file from disk."),
                _AssistantMessage(
                    tool_calls=[
                        _ToolCall(function={"name": "weather_get_forecast"}),
                        _ToolCall(function={"name": "calendar_create_event"}),
                    ]
                ),
            ],
            cycle_index=1,
        )

        self.assertEqual(selection.active_tool_names, ("calendar_create_event",))
        self.assertEqual(selection.retained_tool_names, ("calendar_create_event",))

    def test_invalid_registered_condition_is_rejected(self):
        with self.assertRaises(DynamicSelectorConfigurationError):
            build_registered_dynamic_selector(selector_id="unknown", tool_budget=10)
        with self.assertRaises(DynamicSelectorConfigurationError):
            build_registered_dynamic_selector(
                selector_id=DYNAMIC_STATELESS_SEMANTIC_SELECTOR_ID,
                tool_budget=0,
            )


if __name__ == "__main__":
    unittest.main()
