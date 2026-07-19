"""Execute one dynamic MCP-Atlas condition and write raw/safe separated outputs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from mcp_completion.dynamic_results import (
    DynamicCondition,
    build_dynamic_payload,
    build_evaluator_result_row,
    build_safe_manifest,
)


SOURCE_PIN = "0f307af813334c5174dc0b560c29ce3d5828ee50"


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_one_row(input_csv: Path, task_id: str) -> dict[str, str]:
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("TASK") == task_id:
                return row
    raise ValueError(f"task {task_id!r} was not found in {input_csv}")


def _parse_active_names(value: str | None, source_file: Path | None) -> tuple[str, ...] | None:
    if value is not None and source_file is not None:
        raise ValueError("choose only one of --active-tool-names or --active-tool-names-file")
    if source_file is not None:
        value = source_file.read_text(encoding="ascii")
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(name, str) for name in decoded):
        raise ValueError("--active-tool-names must be a JSON string array")
    return tuple(decoded)


def _write_one_csv_row(path: Path, row: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--active-tool-names")
    parser.add_argument("--active-tool-names-file", type=Path)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--server-url", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    args = parser.parse_args()

    condition = DynamicCondition(
        condition_id=args.condition_id,
        active_tool_names=_parse_active_names(
            args.active_tool_names, args.active_tool_names_file
        ),
    )
    source_row = _load_one_row(args.input, args.task_id)
    payload = build_dynamic_payload(
        source_row,
        model=args.model,
        condition=condition,
        max_turns=args.max_turns,
    )

    started = time.monotonic()
    endpoint_response = _post_json(
        f"{args.server_url.rstrip('/')}/v2/mcp_eval/run_agent_dynamic",
        payload,
        args.timeout_seconds,
    )
    result_row = build_evaluator_result_row(
        source_row,
        endpoint_response=endpoint_response,
        condition=condition,
        model=args.model,
        elapsed_seconds=time.monotonic() - started,
        attempts=1,
    )
    raw_bytes = _write_one_csv_row(args.raw_output, result_row)
    safe_manifest = build_safe_manifest(
        source_pin=SOURCE_PIN,
        model=args.model,
        condition=condition,
        endpoint_response=endpoint_response,
        raw_result_bytes=raw_bytes,
    )
    args.safe_output.parent.mkdir(parents=True, exist_ok=True)
    args.safe_output.write_text(
        json.dumps(safe_manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(safe_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
