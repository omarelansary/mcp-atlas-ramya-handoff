"""Capture a P1-025 unfiltered MCP-Atlas runtime inventory without an LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import httpx

# Direct script execution places only this ``scripts`` directory on sys.path.
# Keep the documented ``python scripts/...`` command independent of installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_completion.p1_025_inventory import (
    P1025InventoryError,
    build_safe_inventory_manifest,
    canonical_json,
    qualify_runtime_tools,
)


def _validate_raw_output_dir(raw_output_dir: Path) -> Path:
    """Keep unreviewed raw schemas under an already ignored result directory."""

    resolved = raw_output_dir.resolve()
    allowed_roots = (
        (PROJECT_ROOT / "completion_results").resolve(),
        (PROJECT_ROOT / "evaluation_results").resolve(),
    )
    if not any(
        resolved == allowed_root or allowed_root in resolved.parents
        for allowed_root in allowed_roots
    ):
        raise P1025InventoryError(
            "--raw-output-dir must be inside services/mcp_eval/completion_results "
            "or services/mcp_eval/evaluation_results; raw schemas must remain ignored"
        )
    return resolved


async def _get_json(client: httpx.AsyncClient, url: str, *, method: str) -> Any:
    response = await getattr(client, method)(url)
    response.raise_for_status()
    return response.json()


async def _run(args: argparse.Namespace) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    base_url = args.sandbox_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        status_payload = await _get_json(client, f"{base_url}/enabled-servers", method="get")
        server_status = _parse_server_status(status_payload)
        configured_server_ids = tuple(sorted(server_status))
        raw_snapshots: list[list[dict[str, Any]]] = []
        qualified_snapshots: list[list[dict[str, Any]]] = []
        for _ in range(args.snapshot_count):
            raw_payload = await _get_json(client, f"{base_url}/list-tools", method="post")
            if not isinstance(raw_payload, list):
                raise P1025InventoryError("/list-tools must return a JSON list")
            raw_snapshot = json.loads(canonical_json(raw_payload))
            raw_snapshots.append(raw_snapshot)
            qualified_snapshots.append(
                qualify_runtime_tools(
                    raw_snapshot,
                    configured_server_ids=configured_server_ids,
                )
            )

    manifest = build_safe_inventory_manifest(
        source_repository_commit=args.source_repository_commit,
        dataset_revision=args.dataset_revision,
        sandbox_url=base_url,
        expected_server_count=args.expected_server_count,
        server_status=server_status,
        raw_snapshots=raw_snapshots,
        qualified_snapshots=qualified_snapshots,
    )
    return raw_snapshots, manifest


def _parse_server_status(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("servers"), list):
        raise P1025InventoryError("/enabled-servers must return a servers list")
    statuses: dict[str, str] = {}
    for entry in payload["servers"]:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not entry[0]
            or not isinstance(entry[1], str)
            or not entry[1]
        ):
            raise P1025InventoryError("/enabled-servers entries must be [server_id, status]")
        if entry[0] in statuses:
            raise P1025InventoryError(f"duplicate configured server {entry[0]!r}")
        statuses[entry[0]] = entry[1]
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture two unfiltered no-provider MCP-Atlas inventory snapshots."
    )
    parser.add_argument("--sandbox-url", default="http://127.0.0.1:1984")
    parser.add_argument(
        "--source-repository-commit",
        default="0f307af813334c5174dc0b560c29ce3d5828ee50",
    )
    parser.add_argument(
        "--dataset-revision",
        default="8c563b55d7c967755f474299848049834d624617",
    )
    parser.add_argument("--expected-server-count", type=int, default=20)
    parser.add_argument("--snapshot-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        required=True,
        help="Ignored/private directory for unfiltered runtime tool snapshots.",
    )
    parser.add_argument(
        "--safe-manifest",
        type=Path,
        required=True,
        help="Reviewed aggregate manifest path; it contains no raw tool records.",
    )
    args = parser.parse_args()
    if args.snapshot_count < 2:
        parser.error("--snapshot-count must be at least 2")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    try:
        args.raw_output_dir = _validate_raw_output_dir(args.raw_output_dir)
    except P1025InventoryError as exc:
        parser.error(str(exc))

    raw_snapshots, manifest = asyncio.run(_run(args))
    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    for index, snapshot in enumerate(raw_snapshots, start=1):
        (args.raw_output_dir / f"raw_tools_snapshot_{index}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    args.safe_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.safe_manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
