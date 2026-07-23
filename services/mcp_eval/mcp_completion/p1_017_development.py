"""Frozen P1-017 development-grid planning and request construction."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dynamic_results import DynamicCondition, build_dynamic_payload


SOURCE_PIN = "0f307af813334c5174dc0b560c29ce3d5828ee50"
SOURCE_TASK_FILE_SHA256 = "d3bbb0119d1c822cdfad158316052c6a96e332d6b6b25f80f501dce6826b6929"
CORE_RELEASE_TAG = "active-registry-core/v0.1.0"
CORE_COMMIT = "e749c6cfa24c1235a548a18d6a1e7f261abbb252"
CONTAINER_IMAGE = (
    "ghcr.io/scaleapi/mcp-atlas:1.2.5"
    "@sha256:415a532f1aeae911fbe2d337cde0657c345a937fab295989f6ef8c70c09c740f"
)
SERVER_MODE = "default-no-key"
EVALUATOR_PASS_THRESHOLD = 0.75
DEVELOPMENT_TASK_IDS = (
    "6888e207a34beb25cfedda3b",
    "688ba1b3e95696e72dd93e8a",
    "68993ef3cf3e953b8ab83fa9",
    "689af4e653c3905e7b5b25b8",
    "689bd255c0422b257e7dfcc5",
)
CANDIDATE_BUDGETS = (10, 20)
REPETITIONS = (1, 2)
SELECTOR_IDS = (
    "dynamic_full",
    "dynamic_stateless_semantic",
    "dynamic_called_tool_retention",
)
EXECUTION_MANIFEST_REQUIRED_FIELDS = (
    "source_pin",
    "source_task_file_sha256",
    "core_release_tag",
    "core_commit",
    "adapter_commit",
    "container_image",
    "server_mode",
    "completion_model",
    "evaluator_model",
    "evaluator_pass_threshold",
    "max_turns",
    "provider_parameters",
)
_FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "activeToolNames",
        "enabledTools",
        "ENABLED_TOOLS",
        "TRAJECTORY",
        "GTFA_CLAIMS",
        "coverage",
        "evaluator",
    }
)
_CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)


@dataclass(frozen=True)
class P1017DevelopmentRun:
    """One completion request in the predeclared development grid."""

    task_id: str
    selector_id: str
    tool_budget: int
    repetition: int

    @property
    def condition_id(self) -> str:
        return f"{self.selector_id}_b{self.tool_budget}_r{self.repetition}"

    @property
    def result_key(self) -> str:
        return f"{self.task_id}__{self.condition_id}"


def load_frozen_development_rows(input_csv: Path) -> dict[str, dict[str, str]]:
    """Verify the frozen source file and return only its five development rows."""

    file_hash = hashlib.sha256(input_csv.read_bytes()).hexdigest()
    if file_hash != SOURCE_TASK_FILE_SHA256:
        raise ValueError(
            "P1-017 requires the frozen sample_tasks.csv SHA-256; "
            f"expected {SOURCE_TASK_FILE_SHA256}, got {file_hash}"
        )

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    indexed_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        task_id = row.get("TASK")
        if not task_id:
            raise ValueError("source task row is missing TASK")
        if task_id in indexed_rows:
            raise ValueError(f"source task file contains duplicate task {task_id!r}")
        indexed_rows[task_id] = row

    missing = [task_id for task_id in DEVELOPMENT_TASK_IDS if task_id not in indexed_rows]
    if missing:
        raise ValueError(f"frozen development tasks are missing: {missing!r}")
    return {task_id: indexed_rows[task_id] for task_id in DEVELOPMENT_TASK_IDS}


def build_development_grid() -> tuple[P1017DevelopmentRun, ...]:
    """Return all 60 frozen task/condition/budget/repetition requests."""

    return tuple(
        P1017DevelopmentRun(
            task_id=task_id,
            selector_id=selector_id,
            tool_budget=tool_budget,
            repetition=repetition,
        )
        for tool_budget in CANDIDATE_BUDGETS
        for selector_id in SELECTOR_IDS
        for repetition in REPETITIONS
        for task_id in DEVELOPMENT_TASK_IDS
    )


def build_registered_condition(run: P1017DevelopmentRun) -> DynamicCondition:
    """Construct a registered-only endpoint condition for one planned request."""

    if run.selector_id not in SELECTOR_IDS:
        raise ValueError(f"unknown P1-017 selector {run.selector_id!r}")
    if run.tool_budget not in CANDIDATE_BUDGETS:
        raise ValueError(f"unexpected P1-017 budget {run.tool_budget!r}")
    if run.repetition not in REPETITIONS:
        raise ValueError(f"unexpected P1-017 repetition {run.repetition!r}")
    return DynamicCondition(
        condition_id=run.condition_id,
        active_tool_names=None,
        selector_id=run.selector_id,
        tool_budget=run.tool_budget,
    )


def build_registered_request_payload(
    source_row: Mapping[str, Any],
    *,
    run: P1017DevelopmentRun,
    model: str,
    max_turns: int,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one selector-safe dynamic request without source/evaluator fields."""

    payload = build_dynamic_payload(
        source_row,
        model=model,
        condition=build_registered_condition(run),
        max_turns=max_turns,
        extra_body=extra_body,
    )
    leaked = sorted(_FORBIDDEN_PAYLOAD_FIELDS.intersection(payload))
    if leaked:
        raise AssertionError(f"forbidden P1-017 request fields: {leaked!r}")
    if payload.get("selectorId") != run.selector_id:
        raise AssertionError("registered selector ID was not preserved")
    if payload.get("toolBudget") != run.tool_budget:
        raise AssertionError("registered tool budget was not preserved")
    return payload


def build_preflight_manifest(
    *,
    model: str,
    max_turns: int,
    runs: Sequence[P1017DevelopmentRun],
) -> dict[str, Any]:
    """Return a safe, prompt-free record of the frozen execution plan."""

    return {
        "p1_record": "P1-017-T2-development",
        "scope": "frozen dynamic MCP-Atlas development grid; no execution or score",
        "source_pin": SOURCE_PIN,
        "source_task_file_sha256": SOURCE_TASK_FILE_SHA256,
        "development_task_ids": list(DEVELOPMENT_TASK_IDS),
        "selector_ids": list(SELECTOR_IDS),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "repetitions": list(REPETITIONS),
        "planned_request_count": len(runs),
        "model": model,
        "max_turns": max_turns,
    }


def load_execution_manifest(
    path: Path,
    *,
    model: str,
    max_turns: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and validate a credential-free configuration required for execution."""

    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("P1-017 execution manifest must be a JSON object")
    _validate_execution_manifest(decoded, model=model, max_turns=max_turns)
    raw_bytes = path.read_bytes()
    return decoded, dict(decoded["provider_parameters"]), hashlib.sha256(raw_bytes).hexdigest()


def _validate_execution_manifest(
    manifest: Mapping[str, Any],
    *,
    model: str,
    max_turns: int,
) -> None:
    missing = [field for field in EXECUTION_MANIFEST_REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"P1-017 execution manifest is missing fields: {missing!r}")
    if manifest["source_pin"] != SOURCE_PIN:
        raise ValueError("execution manifest source_pin does not match the frozen P1-017 source")
    if manifest["source_task_file_sha256"] != SOURCE_TASK_FILE_SHA256:
        raise ValueError("execution manifest source task hash does not match P1-017")
    if manifest["core_release_tag"] != CORE_RELEASE_TAG:
        raise ValueError("execution manifest core_release_tag does not match P1-017")
    if manifest["core_commit"] != CORE_COMMIT:
        raise ValueError("execution manifest core_commit does not match P1-017")
    if manifest["container_image"] != CONTAINER_IMAGE:
        raise ValueError("execution manifest container_image does not match P1-017")
    if manifest["server_mode"] != SERVER_MODE:
        raise ValueError("execution manifest server_mode does not match P1-017")
    if manifest["evaluator_pass_threshold"] != EVALUATOR_PASS_THRESHOLD:
        raise ValueError(
            "execution manifest evaluator_pass_threshold does not match P1-017"
        )
    if manifest["completion_model"] != model:
        raise ValueError("execution manifest completion_model does not match --model")
    if manifest["max_turns"] != max_turns:
        raise ValueError("execution manifest max_turns does not match --max-turns")
    if not isinstance(manifest["provider_parameters"], Mapping):
        raise ValueError("execution manifest provider_parameters must be an object")
    if _contains_credential_field(manifest):
        raise ValueError("execution manifest must not contain credentials")
    for field in EXECUTION_MANIFEST_REQUIRED_FIELDS:
        if field == "provider_parameters":
            continue
        if not manifest[field]:
            raise ValueError(f"execution manifest field {field!r} must be nonempty")


def _contains_credential_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = key.lower().replace("-", "_") if isinstance(key, str) else ""
            if normalized_key in _CREDENTIAL_FIELD_NAMES:
                return True
            if _contains_credential_field(nested_value):
                return True
    elif isinstance(value, list):
        return any(_contains_credential_field(item) for item in value)
    return False
