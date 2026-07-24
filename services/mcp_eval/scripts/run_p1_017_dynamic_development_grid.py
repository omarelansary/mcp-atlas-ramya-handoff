"""Run a frozen P1-017 dynamic MCP-Atlas cohort grid.

The command is a dry run unless ``--execute`` is supplied. Completion rows and
per-run manifests remain under an ignored output directory because they contain
source ground truth and model/evaluator inputs. Scoring remains a separate,
explicit MCP-Atlas evaluator invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_completion.dynamic_results import (
    build_failed_evaluator_result_row,
    build_safe_failure_manifest,
    build_evaluator_result_row,
    build_safe_manifest,
)
from mcp_completion.p1_017_development import (
    SOURCE_PIN,
    P1017DevelopmentRun,
    build_development_grid,
    build_heldout_grid,
    build_heldout_preflight_manifest,
    build_preflight_manifest,
    build_registered_condition,
    build_registered_request_payload,
    load_execution_manifest,
    load_frozen_development_rows,
    load_frozen_heldout_rows,
)


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DEVELOPMENT_RECORD = "P1-017-T2-development"
_DEVELOPMENT_SCOPE = "dynamic MCP-Atlas development completion; no evaluator result"
_HELDOUT_RECORD = "P1-017-T3-heldout"
_HELDOUT_SCOPE = "dynamic MCP-Atlas held-out completion; no evaluator result"


def _classify_http_failure(error: HTTPError, response_body: bytes) -> str:
    """Classify a prompt-free failure code returned by the dynamic endpoint."""

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    failure_code = detail.get("failure_code") if isinstance(detail, dict) else None
    if failure_code in {
        "hidden_tool_request",
        "max_turns_exhausted",
        "mcp_tool_call_timeout",
        "mcp_tool_execution_error",
        "dynamic_contract_error",
        "dynamic_host_error",
    }:
        return failure_code

    if error.code == 500 and b"model requested hidden tool" in response_body:
        return "hidden_tool_request"
    return "completion_http_error"


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _append_raw_row(path: Path, row: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    serialized_row = StringIO(newline="")
    csv.DictWriter(serialized_row, fieldnames=list(row)).writerow(row)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return serialized_row.getvalue().encode("utf-8")


def _record_completion_failure(
    *,
    source_row: dict[str, str],
    condition: Any,
    model: str,
    elapsed_seconds: float,
    raw_csv: Path,
    p1_record: str,
    scope: str,
    run: P1017DevelopmentRun,
    failure_kind: str,
    http_status: int | None,
) -> dict[str, Any]:
    """Write one evaluator-compatible failure row without endpoint details."""

    raw_row = build_failed_evaluator_result_row(
        source_row,
        condition=condition,
        model=model,
        elapsed_seconds=elapsed_seconds,
        attempts=1,
        failure_kind=failure_kind,
        http_status=http_status,
    )
    raw_row["condition_id"] = condition.condition_id
    raw_row["repetition"] = run.repetition
    raw_row["failure_kind"] = failure_kind
    raw_row["http_status"] = "" if http_status is None else http_status
    raw_bytes = _append_raw_row(raw_csv, raw_row)
    manifest = build_safe_failure_manifest(
        source_pin=SOURCE_PIN,
        model=model,
        condition=condition,
        raw_result_bytes=raw_bytes,
        failure_kind=failure_kind,
        http_status=http_status,
        p1_record=p1_record,
        scope=scope,
    )
    manifest["result_key"] = run.result_key
    manifest["repetition"] = run.repetition
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cohort",
        choices=("development", "heldout"),
        default="development",
        help="Frozen P1-017 cohort; heldout uses only the T2-selected B=10.",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:3000")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--raw-output-dir", type=Path, default=Path("completion_results"))
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        help=(
            "Credential-free JSON that freezes the execution configuration; "
            "required with --execute."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send the frozen cohort completion requests; omitted means dry-run only.",
    )
    return parser.parse_args()


def _run_directory(raw_output_dir: Path, run_id: str, phase_directory: str) -> Path:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("--run-id must use only letters, digits, '.', '_', or '-'")
    return raw_output_dir / phase_directory / run_id


def _execute_run(
    *,
    run: P1017DevelopmentRun,
    source_row: dict[str, str],
    model: str,
    max_turns: int,
    server_url: str,
    timeout_seconds: float,
    raw_csv: Path,
    extra_body: dict[str, Any],
    p1_record: str,
    scope: str,
) -> dict[str, Any]:
    condition = build_registered_condition(run)
    payload = build_registered_request_payload(
        source_row,
        run=run,
        model=model,
        max_turns=max_turns,
        extra_body=extra_body,
    )
    started = time.monotonic()
    try:
        endpoint_response = _post_json(
            f"{server_url.rstrip('/')}/v2/mcp_eval/run_agent_dynamic",
            payload,
            timeout_seconds,
        )
    except HTTPError as error:
        failure_kind = _classify_http_failure(error, error.read())
        return _record_completion_failure(
            source_row=source_row,
            condition=condition,
            model=model,
            elapsed_seconds=time.monotonic() - started,
            raw_csv=raw_csv,
            p1_record=p1_record,
            scope=scope,
            run=run,
            failure_kind=failure_kind,
            http_status=error.code,
        )
    except TimeoutError:
        # The approved recovery records only the frozen local request timeout.
        # Connection and other transport errors remain hard stops.
        return _record_completion_failure(
            source_row=source_row,
            condition=condition,
            model=model,
            elapsed_seconds=time.monotonic() - started,
            raw_csv=raw_csv,
            p1_record=p1_record,
            scope=scope,
            run=run,
            failure_kind="completion_timeout",
            http_status=None,
        )
    else:
        raw_row = build_evaluator_result_row(
            source_row,
            endpoint_response=endpoint_response,
            condition=condition,
            model=model,
            elapsed_seconds=time.monotonic() - started,
            attempts=1,
        )
        raw_row["condition_id"] = condition.condition_id
        raw_row["repetition"] = run.repetition
        raw_row["failure_kind"] = ""
        raw_row["http_status"] = ""
        raw_bytes = _append_raw_row(raw_csv, raw_row)
        manifest = build_safe_manifest(
            source_pin=SOURCE_PIN,
            model=model,
            condition=condition,
            endpoint_response=endpoint_response,
            raw_result_bytes=raw_bytes,
            p1_record=p1_record,
            scope=scope,
        )
    manifest["result_key"] = run.result_key
    manifest["repetition"] = run.repetition
    return manifest


def main() -> int:
    args = _parse_args()
    if args.cohort == "development":
        cohort_rows = load_frozen_development_rows(args.input)
        runs = build_development_grid()
        preflight = build_preflight_manifest(
            model=args.model,
            max_turns=args.max_turns,
            runs=runs,
        )
        phase_directory = "p1_017_t2"
        p1_record = _DEVELOPMENT_RECORD
        scope = _DEVELOPMENT_SCOPE
    else:
        cohort_rows = load_frozen_heldout_rows(args.input)
        runs = build_heldout_grid()
        preflight = build_heldout_preflight_manifest(
            model=args.model,
            max_turns=args.max_turns,
            runs=runs,
        )
        phase_directory = "p1_017_t3"
        p1_record = _HELDOUT_RECORD
        scope = _HELDOUT_SCOPE

    run_directory = _run_directory(args.raw_output_dir, args.run_id, phase_directory)
    preflight["run_id"] = args.run_id

    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    if args.execution_manifest is None:
        raise ValueError("--execute requires --execution-manifest")
    execution_manifest, extra_body, manifest_hash = load_execution_manifest(
        args.execution_manifest,
        model=args.model,
        max_turns=args.max_turns,
    )
    preflight["execution_manifest_sha256"] = manifest_hash
    preflight["execution_configuration"] = {
        key: execution_manifest[key]
        for key in (
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
    }

    if run_directory.exists():
        raise FileExistsError(
            f"refusing to reuse existing run directory {run_directory}; choose a new run ID"
        )
    run_directory.mkdir(parents=True)
    raw_csv = run_directory / "completion_rows.csv"
    completed: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        manifest = _execute_run(
            run=run,
            source_row=cohort_rows[run.task_id],
            model=args.model,
            max_turns=args.max_turns,
            server_url=args.server_url,
            timeout_seconds=args.timeout_seconds,
            raw_csv=raw_csv,
            extra_body=extra_body,
            p1_record=p1_record,
            scope=scope,
        )
        completed.append(manifest)
        print(f"completed {index}/{len(runs)}: {run.result_key}")

    preflight["execution_completed"] = True
    preflight["completed_request_count"] = len(completed)
    preflight["raw_completion_rows_path"] = str(raw_csv)
    preflight["runs"] = completed
    (run_directory / "safe_completion_manifest.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
