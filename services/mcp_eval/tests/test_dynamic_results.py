"""P1-016 dynamic result-format tests without endpoint or provider calls."""

from __future__ import annotations

import unittest

from mcp_completion.dynamic_results import (
    DynamicCondition,
    build_dynamic_payload,
    build_evaluator_result_row,
    build_safe_manifest,
)


def _source_row():
    return {
        "TASK": "task-1",
        "PROMPT": "PROMPT_MARKER",
        "TRAJECTORY": "TRAJECTORY_MARKER",
        "GTFA_CLAIMS": "CLAIMS_MARKER",
        "ENABLED_TOOLS": '["files_read"]',
    }


def _endpoint_response():
    return {
        "outputs": [
            {
                "type": "message",
                "data": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "files_read", "arguments": "{}"},
                        }
                    ],
                },
            },
            {
                "type": "message",
                "data": {"role": "tool", "tool_call_id": "call-1", "content": "RAW_TOOL_RESULT"},
            },
            {"type": "message", "data": {"role": "assistant", "content": "ANSWER_MARKER"}},
        ],
        "dynamic_trace": {
            "cycles": [
                {
                    "cycle_index": 0,
                    "raw_tool_names": ["files_read"],
                    "raw_tool_hash": "raw-hash",
                    "active_tool_names": ["files_read"],
                    "provider_tools_hash": "provider-hash",
                }
            ],
            "model_final_text_present": True,
        },
    }


class DynamicResultsTests(unittest.TestCase):
    def test_payload_excludes_evaluator_columns(self):
        payload = build_dynamic_payload(
            _source_row(),
            model="fake/model",
            condition=DynamicCondition("selected", ("files_read",)),
            max_turns=2,
        )
        serialized = str(payload)
        self.assertEqual(payload["activeToolNames"], ["files_read"])
        self.assertNotIn("TRAJECTORY", payload)
        self.assertNotIn("GTFA_CLAIMS", payload)
        self.assertNotIn("TRAJECTORY_MARKER", serialized)
        self.assertNotIn("CLAIMS_MARKER", serialized)

    def test_result_row_is_evaluator_compatible(self):
        row = build_evaluator_result_row(
            _source_row(),
            endpoint_response=_endpoint_response(),
            condition=DynamicCondition("selected", ("files_read",)),
            model="fake/model",
            elapsed_seconds=1.5,
            attempts=1,
        )
        for column in ("TASK", "PROMPT", "TRAJECTORY", "GTFA_CLAIMS", "script_model_response"):
            self.assertIn(column, row)
        self.assertEqual(row["script_model_response"], "ANSWER_MARKER")
        self.assertEqual(row["exposure_mode"], "dynamic-selected")

    def test_safe_manifest_excludes_raw_source_and_response_content(self):
        manifest = build_safe_manifest(
            source_pin="pin",
            model="fake/model",
            condition=DynamicCondition("selected", ("files_read",)),
            endpoint_response=_endpoint_response(),
            raw_result_bytes=b"raw-csv",
        )
        serialized = str(manifest)
        self.assertEqual(manifest["dynamic_cycle_count"], 1)
        self.assertNotIn("PROMPT_MARKER", serialized)
        self.assertNotIn("TRAJECTORY_MARKER", serialized)
        self.assertNotIn("CLAIMS_MARKER", serialized)
        self.assertNotIn("ANSWER_MARKER", serialized)
        self.assertNotIn("RAW_TOOL_RESULT", serialized)


if __name__ == "__main__":
    unittest.main()
