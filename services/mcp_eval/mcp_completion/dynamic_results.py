"""Evaluator-compatible result records for the separate P1-016 dynamic route."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


EVALUATOR_REQUIRED_SOURCE_COLUMNS = (
    "TASK",
    "PROMPT",
    "TRAJECTORY",
    "GTFA_CLAIMS",
)


@dataclass(frozen=True)
class DynamicCondition:
    """Predeclared P1-016 explicit-set or P1-017 registered condition."""

    condition_id: str
    active_tool_names: tuple[str, ...] | None
    selector_id: str | None = None
    tool_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be nonempty")
        if self.active_tool_names is not None:
            if len(set(self.active_tool_names)) != len(self.active_tool_names):
                raise ValueError("active_tool_names must not contain duplicates")
            if any(not name for name in self.active_tool_names):
                raise ValueError("active_tool_names must not contain empty names")
        if self.selector_id is not None:
            if self.active_tool_names is not None:
                raise ValueError("registered conditions must not set active_tool_names")
            if not self.selector_id:
                raise ValueError("selector_id must be nonempty when provided")
            if not isinstance(self.tool_budget, int) or self.tool_budget < 1:
                raise ValueError("registered conditions require a positive tool_budget")
        elif self.tool_budget is not None:
            raise ValueError("tool_budget requires selector_id")


def build_dynamic_payload(
    source_row: Mapping[str, Any],
    *,
    model: str,
    condition: DynamicCondition,
    max_turns: int,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the endpoint payload without evaluator-only source columns."""

    _require_source_columns(source_row)
    if not model:
        raise ValueError("model must be nonempty")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": source_row["PROMPT"]}],
        "maxTurns": max_turns,
    }
    if condition.selector_id is not None:
        payload["selectorId"] = condition.selector_id
        payload["toolBudget"] = condition.tool_budget
    elif condition.active_tool_names is not None:
        payload["activeToolNames"] = list(condition.active_tool_names)
    if extra_body:
        payload["extraBody"] = copy.deepcopy(dict(extra_body))
    return payload


def build_evaluator_result_row(
    source_row: Mapping[str, Any],
    *,
    endpoint_response: Mapping[str, Any],
    condition: DynamicCondition,
    model: str,
    elapsed_seconds: float,
    attempts: int,
) -> dict[str, Any]:
    """Build an ignored CSV row compatible with MCP-Atlas's existing scorer."""

    _require_source_columns(source_row)
    outputs = endpoint_response.get("outputs")
    trace = endpoint_response.get("dynamic_trace")
    if not isinstance(outputs, list) or not isinstance(trace, dict):
        raise ValueError("dynamic endpoint response must contain outputs and dynamic_trace")

    return {
        "TASK": source_row["TASK"],
        "PROMPT": source_row["PROMPT"],
        "TRAJECTORY": source_row["TRAJECTORY"],
        "GTFA_CLAIMS": source_row["GTFA_CLAIMS"],
        "ENABLED_TOOLS": source_row.get("ENABLED_TOOLS", ""),
        "script_model_response": extract_final_response(outputs),
        "raw_conversation_history": json.dumps(outputs, ensure_ascii=True),
        "trajectory": json.dumps(extract_tool_calls(outputs), ensure_ascii=True),
        "errors": "[]",
        "trajectory_time": elapsed_seconds,
        "num_retry": attempts,
        "exposure_mode": _exposure_mode(condition),
        "configured_active_tool_names": json.dumps(
            list(condition.active_tool_names or ()), ensure_ascii=True
        ),
        "selector_id": condition.selector_id or "",
        "tool_budget": condition.tool_budget if condition.tool_budget is not None else "",
        "dynamic_trace": json.dumps(trace, ensure_ascii=True, sort_keys=True),
        "model_id": model,
    }


def build_safe_manifest(
    *,
    source_pin: str,
    model: str,
    condition: DynamicCondition,
    endpoint_response: Mapping[str, Any],
    raw_result_bytes: bytes,
    p1_record: str = "P1-016-T2-preflight",
    scope: str = "dynamic MCP-Atlas result-format compatibility; not a task-preservation result",
) -> dict[str, Any]:
    """Create a tracked-safe summary without evaluator data or raw output."""

    trace = endpoint_response.get("dynamic_trace")
    if not isinstance(trace, dict):
        raise ValueError("dynamic endpoint response lacks dynamic_trace")
    cycles = trace.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError("dynamic_trace.cycles must be a list")
    return {
        "p1_record": p1_record,
        "scope": scope,
        "source_pin": source_pin,
        "condition_id": condition.condition_id,
        "model": model,
        "exposure_mode": _exposure_mode(condition),
        "configured_active_tool_names": list(condition.active_tool_names or ()),
        "selector_id": condition.selector_id,
        "tool_budget": condition.tool_budget,
        "dynamic_cycle_count": len(cycles),
        "dynamic_cycles": copy.deepcopy(cycles),
        "raw_result_sha256": hashlib.sha256(raw_result_bytes).hexdigest(),
        "success": True,
    }


def build_failed_evaluator_result_row(
    source_row: Mapping[str, Any],
    *,
    condition: DynamicCondition,
    model: str,
    elapsed_seconds: float,
    attempts: int,
    failure_kind: str,
    http_status: int,
) -> dict[str, Any]:
    """Build an evaluator-compatible row for a completion endpoint failure."""

    _require_source_columns(source_row)
    if not failure_kind:
        raise ValueError("failure_kind must be nonempty")
    return {
        "TASK": source_row["TASK"],
        "PROMPT": source_row["PROMPT"],
        "TRAJECTORY": source_row["TRAJECTORY"],
        "GTFA_CLAIMS": source_row["GTFA_CLAIMS"],
        "ENABLED_TOOLS": source_row.get("ENABLED_TOOLS", ""),
        "script_model_response": "",
        "raw_conversation_history": "[]",
        "trajectory": "[]",
        "errors": json.dumps(
            [{"kind": failure_kind, "http_status": http_status}],
            ensure_ascii=True,
            sort_keys=True,
        ),
        "trajectory_time": elapsed_seconds,
        "num_retry": attempts,
        "exposure_mode": _exposure_mode(condition),
        "configured_active_tool_names": json.dumps(
            list(condition.active_tool_names or ()), ensure_ascii=True
        ),
        "selector_id": condition.selector_id or "",
        "tool_budget": condition.tool_budget if condition.tool_budget is not None else "",
        "dynamic_trace": json.dumps(
            {"failure_kind": failure_kind, "http_status": http_status},
            ensure_ascii=True,
            sort_keys=True,
        ),
        "model_id": model,
    }


def build_safe_failure_manifest(
    *,
    source_pin: str,
    model: str,
    condition: DynamicCondition,
    raw_result_bytes: bytes,
    failure_kind: str,
    http_status: int,
    p1_record: str = "P1-016-T2-preflight",
    scope: str = "dynamic MCP-Atlas completion failure; not an evaluator result",
) -> dict[str, Any]:
    """Create a prompt-free safe record for an endpoint completion failure."""

    if not failure_kind:
        raise ValueError("failure_kind must be nonempty")
    return {
        "p1_record": p1_record,
        "scope": scope,
        "source_pin": source_pin,
        "condition_id": condition.condition_id,
        "model": model,
        "exposure_mode": _exposure_mode(condition),
        "configured_active_tool_names": list(condition.active_tool_names or ()),
        "selector_id": condition.selector_id,
        "tool_budget": condition.tool_budget,
        "dynamic_cycle_count": 0,
        "dynamic_cycles": [],
        "raw_result_sha256": hashlib.sha256(raw_result_bytes).hexdigest(),
        "success": False,
        "failure_kind": failure_kind,
        "http_status": http_status,
    }


def extract_final_response(outputs: Sequence[Mapping[str, Any]]) -> str:
    """Match the official runner's assistant-content-first response extraction."""

    for output in reversed(outputs):
        if output.get("type") != "message":
            continue
        data = output.get("data")
        if not isinstance(data, Mapping) or data.get("role") != "assistant":
            continue
        content = data.get("content")
        if isinstance(content, str) and content:
            return content
        tool_calls = data.get("tool_calls")
        if tool_calls:
            return json.dumps(tool_calls, ensure_ascii=True, sort_keys=True)
    return ""


def extract_tool_calls(outputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract tool-call names and parsed arguments for ignored raw CSV use."""

    calls: list[dict[str, Any]] = []
    for output in outputs:
        data = output.get("data") if output.get("type") == "message" else None
        if not isinstance(data, Mapping):
            continue
        for call in data.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, Mapping) else {}
            arguments = function.get("arguments", "{}")
            try:
                parsed_arguments = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                parsed_arguments = {}
            calls.append(
                {
                    "tool_name": function.get("name", ""),
                    "parameters": parsed_arguments,
                }
            )
    return calls


def _require_source_columns(source_row: Mapping[str, Any]) -> None:
    missing = [
        column
        for column in EVALUATOR_REQUIRED_SOURCE_COLUMNS
        if not isinstance(source_row.get(column), str)
    ]
    if missing:
        raise ValueError(f"source row is missing string columns: {missing!r}")


def _exposure_mode(condition: DynamicCondition) -> str:
    if condition.selector_id == "dynamic_full" or (
        condition.selector_id is None and condition.active_tool_names is None
    ):
        return "dynamic-full"
    return "dynamic-selected"
