"""No-provider tests for the P1-025 MCP-Atlas inventory-freeze gate."""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import unittest

from mcp_completion.p1_025_inventory import (
    P1025InventoryError,
    build_safe_inventory_manifest,
    qualify_runtime_tools,
)


def _runtime_tools() -> list[dict]:
    return [
        {
            "name": "filesystem_read_file",
            "description": "Read a file.",
            "inputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        },
        {
            "name": "mcp-code-executor_execute_code",
            "description": "Execute code.",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
        },
    ]


class P1025InventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server_ids = ("filesystem", "mcp-code-executor")
        self.status = {server_id: "OK" for server_id in self.server_ids}
        self.raw = _runtime_tools()
        self.qualified = qualify_runtime_tools(
            self.raw, configured_server_ids=self.server_ids
        )

    def test_qualifies_flattened_namespaces_without_task_data(self) -> None:
        self.assertEqual(
            [(item["server_id"], item["tool_name"]) for item in self.qualified],
            [
                ("filesystem", "filesystem_read_file"),
                ("mcp-code-executor", "mcp-code-executor_execute_code"),
            ],
        )
        self.assertNotIn("trajectory", self.qualified[0])
        self.assertNotIn("enabled_tools", self.qualified[0])

    def test_explicit_server_is_checked_against_runtime_configuration(self) -> None:
        raw = _runtime_tools()
        raw[0]["server"] = "filesystem"
        self.assertEqual(
            qualify_runtime_tools(raw, configured_server_ids=self.server_ids)[0]["server_id"],
            "filesystem",
        )
        raw[0]["server"] = "unconfigured"
        with self.assertRaises(P1025InventoryError):
            qualify_runtime_tools(raw, configured_server_ids=self.server_ids)

    def test_unnamespaced_or_ambiguous_tools_fail_the_gate(self) -> None:
        raw = _runtime_tools()
        raw[0]["name"] = "read_file"
        with self.assertRaises(P1025InventoryError):
            qualify_runtime_tools(raw, configured_server_ids=self.server_ids)
        with self.assertRaises(P1025InventoryError):
            qualify_runtime_tools(
                _runtime_tools(), configured_server_ids=("filesystem", "filesystem_read")
            )

    def test_evaluator_and_task_fields_are_rejected(self) -> None:
        for forbidden_field in (
            "ENABLED_TOOLS",
            "enabledTools",
            "TRAJECTORY",
            "GTFA_CLAIMS",
            "gtfaClaims",
            "task_id",
            "taskId",
        ):
            raw = _runtime_tools()
            raw[0][forbidden_field] = "forbidden"
            with self.subTest(forbidden_field=forbidden_field):
                with self.assertRaises(P1025InventoryError):
                    qualify_runtime_tools(raw, configured_server_ids=self.server_ids)

    def test_manifest_requires_two_identical_snapshots_and_all_servers_online(self) -> None:
        manifest = build_safe_inventory_manifest(
            source_repository_commit="source-pin",
            dataset_revision="dataset-pin",
            sandbox_url="http://127.0.0.1:1984",
            expected_server_count=2,
            server_status=self.status,
            raw_snapshots=(self.raw, copy.deepcopy(self.raw)),
            qualified_snapshots=(self.qualified, copy.deepcopy(self.qualified)),
        )

        self.assertTrue(manifest["success"])
        self.assertEqual(manifest["raw_tool_count"], 2)
        self.assertEqual(manifest["server_count_with_tools"], 2)
        self.assertEqual(
            manifest["tools_per_server_profile"],
            {
                "minimum": 1,
                "median": 1,
                "p95_nearest_rank": 1,
                "maximum": 1,
                "histogram": [{"tools_per_server": 1, "server_count": 2}],
            },
        )
        self.assertTrue(manifest["task_rows_read"] is False)
        self.assertNotIn("tool_names", manifest)

        changed = _runtime_tools()
        changed[0]["description"] = "Different"
        with self.assertRaises(P1025InventoryError):
            build_safe_inventory_manifest(
                source_repository_commit="source-pin",
                dataset_revision="dataset-pin",
                sandbox_url="http://127.0.0.1:1984",
                expected_server_count=2,
                server_status=self.status,
                raw_snapshots=(self.raw, changed),
                qualified_snapshots=(self.qualified, self.qualified),
            )

    def test_manifest_profiles_configured_servers_without_disclosing_names(self) -> None:
        server_ids = ("filesystem", "mcp-code-executor", "empty-a", "empty-b")
        status = {server_id: "OK" for server_id in server_ids}
        manifest = build_safe_inventory_manifest(
            source_repository_commit="source-pin",
            dataset_revision="dataset-pin",
            sandbox_url="http://127.0.0.1:1984",
            expected_server_count=4,
            server_status=status,
            raw_snapshots=(self.raw, copy.deepcopy(self.raw)),
            qualified_snapshots=(self.qualified, copy.deepcopy(self.qualified)),
        )

        self.assertEqual(
            manifest["tools_per_server_profile"],
            {
                "minimum": 0,
                "median": 0.5,
                "p95_nearest_rank": 1,
                "maximum": 1,
                "histogram": [
                    {"tools_per_server": 0, "server_count": 2},
                    {"tools_per_server": 1, "server_count": 2},
                ],
            },
        )
        self.assertNotIn("filesystem", str(manifest["tools_per_server_profile"]))

        offline = dict(self.status)
        offline["filesystem"] = "ERROR_NOT_ONLINE"
        with self.assertRaises(P1025InventoryError):
            build_safe_inventory_manifest(
                source_repository_commit="source-pin",
                dataset_revision="dataset-pin",
                sandbox_url="http://127.0.0.1:1984",
                expected_server_count=2,
                server_status=offline,
                raw_snapshots=(self.raw, copy.deepcopy(self.raw)),
                qualified_snapshots=(self.qualified, copy.deepcopy(self.qualified)),
            )

    def test_cli_rejects_a_raw_directory_outside_ignored_results(self) -> None:
        evaluation_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_p1_025_inventory_freeze.py",
                "--raw-output-dir",
                "untracked_raw_schemas",
                "--safe-manifest",
                "completion_results/p1_025/safe_manifest.json",
            ],
            cwd=evaluation_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("raw schemas must remain ignored", completed.stderr)
