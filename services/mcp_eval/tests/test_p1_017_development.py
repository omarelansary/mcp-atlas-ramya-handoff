"""Offline checks for the frozen P1-017 development-grid runner."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mcp_completion.dynamic_results import build_safe_manifest
from mcp_completion.p1_017_development import (
    CANDIDATE_BUDGETS,
    CONTAINER_IMAGE,
    CORE_COMMIT,
    DEVELOPMENT_TASK_IDS,
    EVALUATOR_PASS_THRESHOLD,
    REPETITIONS,
    SELECTOR_IDS,
    SOURCE_TASK_FILE_SHA256,
    build_development_grid,
    build_preflight_manifest,
    build_registered_condition,
    build_registered_request_payload,
    load_execution_manifest,
    load_frozen_development_rows,
)


class P1017DevelopmentGridTests(unittest.TestCase):
    def test_grid_has_one_run_per_frozen_factor_combination(self):
        runs = build_development_grid()

        self.assertEqual(
            len(runs),
            len(DEVELOPMENT_TASK_IDS)
            * len(SELECTOR_IDS)
            * len(CANDIDATE_BUDGETS)
            * len(REPETITIONS),
        )
        self.assertEqual(len({run.result_key for run in runs}), len(runs))
        self.assertEqual({run.task_id for run in runs}, set(DEVELOPMENT_TASK_IDS))
        self.assertEqual({run.selector_id for run in runs}, set(SELECTOR_IDS))
        self.assertEqual({run.tool_budget for run in runs}, set(CANDIDATE_BUDGETS))
        self.assertEqual({run.repetition for run in runs}, set(REPETITIONS))

    def test_frozen_source_file_returns_only_development_rows(self):
        source_file = Path(__file__).resolve().parents[1] / "sample_tasks.csv"
        rows = load_frozen_development_rows(source_file)

        self.assertEqual(tuple(rows), DEVELOPMENT_TASK_IDS)
        self.assertEqual(len(rows), len(DEVELOPMENT_TASK_IDS))

    def test_registered_payload_excludes_source_and_evaluator_fields(self):
        run = build_development_grid()[0]
        source_row = {
            "TASK": run.task_id,
            "PROMPT": "Read a file.",
            "TRAJECTORY": "evaluator-only",
            "GTFA_CLAIMS": "evaluator-only",
            "ENABLED_TOOLS": "source-only",
        }

        payload = build_registered_request_payload(
            source_row,
            run=run,
            model="fake/model",
            max_turns=20,
        )

        self.assertEqual(payload["selectorId"], run.selector_id)
        self.assertEqual(payload["toolBudget"], run.tool_budget)
        self.assertNotIn("activeToolNames", payload)
        self.assertNotIn("ENABLED_TOOLS", payload)
        self.assertNotIn("TRAJECTORY", payload)
        self.assertNotIn("GTFA_CLAIMS", payload)

    def test_condition_is_registered_and_never_contains_active_names(self):
        condition = build_registered_condition(build_development_grid()[0])

        self.assertIsNone(condition.active_tool_names)
        self.assertIn(condition.selector_id, SELECTOR_IDS)
        self.assertIn(condition.tool_budget, CANDIDATE_BUDGETS)

    def test_preflight_manifest_is_prompt_free(self):
        manifest = build_preflight_manifest(
            model="fake/model",
            max_turns=20,
            runs=build_development_grid(),
        )

        self.assertEqual(manifest["planned_request_count"], 60)
        self.assertEqual(manifest["development_task_ids"], list(DEVELOPMENT_TASK_IDS))
        self.assertNotIn("prompt", " ".join(manifest).lower())

    def test_safe_manifest_uses_the_p1_017_label(self):
        run = build_development_grid()[0]
        manifest = build_safe_manifest(
            source_pin="pin",
            model="fake/model",
            condition=build_registered_condition(run),
            endpoint_response={"dynamic_trace": {"cycles": []}},
            raw_result_bytes=b"ignored raw row",
            p1_record="P1-017-T2-development",
            scope="test scope",
        )

        self.assertEqual(manifest["p1_record"], "P1-017-T2-development")
        self.assertEqual(manifest["scope"], "test scope")

    def test_execution_manifest_requires_frozen_nonsecret_configuration(self):
        with TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "execution.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_pin": "0f307af813334c5174dc0b560c29ce3d5828ee50",
                        "source_task_file_sha256": SOURCE_TASK_FILE_SHA256,
                        "core_release_tag": "active-registry-core/v0.1.0",
                        "core_commit": CORE_COMMIT,
                        "adapter_commit": "candidate",
                        "container_image": CONTAINER_IMAGE,
                        "server_mode": "default-no-key",
                        "completion_model": "fake/model",
                        "evaluator_model": "fake/evaluator",
                        "evaluator_pass_threshold": EVALUATOR_PASS_THRESHOLD,
                        "max_turns": 20,
                        "provider_parameters": {"max_tokens": 1024, "temperature": 0},
                    }
                ),
                encoding="utf-8",
            )

            manifest, provider_parameters, manifest_hash = load_execution_manifest(
                manifest_path,
                model="fake/model",
                max_turns=20,
            )

        self.assertEqual(manifest["evaluator_model"], "fake/evaluator")
        self.assertEqual(provider_parameters, {"max_tokens": 1024, "temperature": 0})
        self.assertEqual(len(manifest_hash), 64)

    def test_execution_manifest_rejects_credentials(self):
        with TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "execution.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_pin": "0f307af813334c5174dc0b560c29ce3d5828ee50",
                        "source_task_file_sha256": SOURCE_TASK_FILE_SHA256,
                        "core_release_tag": "active-registry-core/v0.1.0",
                        "core_commit": CORE_COMMIT,
                        "adapter_commit": "candidate",
                        "container_image": CONTAINER_IMAGE,
                        "server_mode": "default-no-key",
                        "completion_model": "fake/model",
                        "evaluator_model": "fake/evaluator",
                        "evaluator_pass_threshold": EVALUATOR_PASS_THRESHOLD,
                        "max_turns": 20,
                        "provider_parameters": {"api_key": "must-not-appear"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "credentials"):
                load_execution_manifest(
                    manifest_path,
                    model="fake/model",
                    max_turns=20,
                )


if __name__ == "__main__":
    unittest.main()
