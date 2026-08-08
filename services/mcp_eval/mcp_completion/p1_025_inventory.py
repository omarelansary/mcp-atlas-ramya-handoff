"""P1-025 MCP-Atlas inventory-freeze helpers.

The helpers are intentionally source-adapter code. They capture the agent
environment's unfiltered ``/list-tools`` response and qualify each tool with a
server origin for the source-neutral core. They do not read task rows, call a
model, invoke tools, or evaluate task claims.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


class P1025InventoryError(RuntimeError):
    """Raised when a runtime inventory cannot be safely frozen."""


_FORBIDDEN_TOOL_FIELD_TOKENS = frozenset(
    {
        "enabledtools",
        "evaluator",
        "evaluatorlabels",
        "gold",
        "goldlabels",
        "gtfaclaims",
        "referenceanswer",
        "score",
        "scores",
        "task",
        "taskid",
        "trajectory",
        "trajectories",
        "verifier",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with the P1-025 stable representation."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def qualify_runtime_tools(
    raw_tools: Sequence[Mapping[str, Any]], *, configured_server_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Attach one verified server origin to every raw MCP tool.

    MCP-Atlas's HTTP endpoint returns a flattened tool list. When a source tool
    includes a ``server`` field, it is checked against the configured server
    list. Otherwise the adapter accepts only one exact namespace match of the
    form ``<configured-server-id>_<tool-name>``. Ambiguous or unnamespaced
    names fail the runtime gate instead of being guessed.
    """

    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise P1025InventoryError("raw tools must be a sequence of objects")
    server_ids = _validated_server_ids(configured_server_ids)
    qualified: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            raise P1025InventoryError("raw tool entries must be objects")
        tool = _normalise_raw_tool(raw_tool)
        server_id = _resolve_server_id(tool, server_ids)
        identity = (server_id, tool["name"])
        if identity in seen:
            raise P1025InventoryError(
                f"duplicate qualified runtime tool {identity!r}"
            )
        seen.add(identity)
        qualified.append(
            {
                "server_id": server_id,
                "tool_name": tool["name"],
                "description": tool.get("description"),
                "input_schema": tool["inputSchema"],
                "output_schema": tool.get("outputSchema"),
                "title": tool.get("title"),
                "annotations": tool.get("annotations"),
                "meta": tool.get("_meta"),
            }
        )

    if not qualified:
        raise P1025InventoryError("runtime discovery returned no tools")
    return sorted(qualified, key=lambda item: (item["server_id"], item["tool_name"]))


def build_safe_inventory_manifest(
    *,
    source_repository_commit: str,
    dataset_revision: str,
    sandbox_url: str,
    expected_server_count: int,
    server_status: Mapping[str, str],
    raw_snapshots: Sequence[Sequence[Mapping[str, Any]]],
    qualified_snapshots: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate repeated discovery and return a safe aggregate manifest.

    The manifest intentionally excludes raw schemas, descriptions, names,
    prompts, task rows, trajectories, tool outputs, and evaluator data.
    """

    _require_nonempty_string(source_repository_commit, "source_repository_commit")
    _require_nonempty_string(dataset_revision, "dataset_revision")
    _require_nonempty_string(sandbox_url, "sandbox_url")
    if not isinstance(expected_server_count, int) or expected_server_count < 1:
        raise P1025InventoryError("expected_server_count must be a positive integer")
    if len(raw_snapshots) < 2 or len(qualified_snapshots) != len(raw_snapshots):
        raise P1025InventoryError("at least two matching raw/qualified snapshots are required")

    normalized_status = _normalise_server_status(server_status)
    online_servers = sorted(
        server_id
        for server_id, status in normalized_status.items()
        if status == "OK"
    )
    if len(normalized_status) != expected_server_count:
        raise P1025InventoryError(
            f"configured server count {len(normalized_status)} does not match expected "
            f"{expected_server_count}"
        )
    if len(online_servers) != expected_server_count:
        offline = sorted(
            server_id
            for server_id, status in normalized_status.items()
            if status != "OK"
        )
        raise P1025InventoryError(
            f"runtime gate requires all configured servers online; offline={offline!r}"
        )

    raw_hashes = [sha256_json(snapshot) for snapshot in raw_snapshots]
    qualified_hashes = [sha256_json(snapshot) for snapshot in qualified_snapshots]
    if len(set(raw_hashes)) != 1:
        raise P1025InventoryError("repeated raw discovery snapshots differ")
    if len(set(qualified_hashes)) != 1:
        raise P1025InventoryError("repeated qualified inventory snapshots differ")

    inventory = qualified_snapshots[0]
    server_to_tool_names: dict[str, list[str]] = {}
    for tool in inventory:
        if not isinstance(tool, Mapping):
            raise P1025InventoryError("qualified inventory entries must be objects")
        server_id = tool.get("server_id")
        if not isinstance(server_id, str) or not server_id:
            raise P1025InventoryError("qualified inventory is missing a server_id")
        if server_id not in normalized_status:
            raise P1025InventoryError(
                f"qualified runtime tool has unconfigured server {server_id!r}"
            )
        tool_name = tool.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise P1025InventoryError("qualified inventory is missing a tool_name")
        server_to_tool_names.setdefault(server_id, []).append(tool_name)
    server_to_tool_names = {
        server_id: sorted(tool_names)
        for server_id, tool_names in sorted(server_to_tool_names.items())
    }
    tools_per_server_profile = _tools_per_server_profile(
        configured_server_ids=tuple(sorted(normalized_status)),
        server_to_tool_names=server_to_tool_names,
    )

    return {
        "p1_record": "P1-025-R1",
        "scope": "MCP-Atlas unfiltered no-provider runtime inventory capture; not a policy or task result",
        "source_repository_commit": source_repository_commit,
        "dataset_revision": dataset_revision,
        "sandbox_url": sandbox_url,
        "provider_used": False,
        "evaluator_used": False,
        "task_rows_read": False,
        "snapshot_count": len(raw_snapshots),
        "configured_server_count": len(normalized_status),
        "online_server_count": len(online_servers),
        "raw_tool_count": len(raw_snapshots[0]),
        "qualified_tool_count": len(inventory),
        "server_count_with_tools": len(server_to_tool_names),
        "tools_per_server_profile": tools_per_server_profile,
        "raw_snapshot_sha256": raw_hashes[0],
        "qualified_inventory_sha256": qualified_hashes[0],
        "server_tool_mapping_sha256": sha256_json(server_to_tool_names),
        "success": True,
    }


def _tools_per_server_profile(
    *,
    configured_server_ids: Sequence[str],
    server_to_tool_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Return anonymous fanout statistics without publishing server/tool names."""

    counts = sorted(
        len(server_to_tool_names.get(server_id, ()))
        for server_id in configured_server_ids
    )
    if not counts:
        raise P1025InventoryError("tools-per-server profile requires configured servers")
    midpoint = len(counts) // 2
    median = (
        counts[midpoint]
        if len(counts) % 2
        else (counts[midpoint - 1] + counts[midpoint]) / 2
    )
    percentile_index = max(0, (95 * len(counts) + 99) // 100 - 1)
    histogram: list[dict[str, int]] = []
    for count in sorted(set(counts)):
        histogram.append(
            {
                "tools_per_server": count,
                "server_count": counts.count(count),
            }
        )
    return {
        "minimum": counts[0],
        "median": median,
        "p95_nearest_rank": counts[percentile_index],
        "maximum": counts[-1],
        "histogram": histogram,
    }


def _normalise_raw_tool(raw_tool: Mapping[str, Any]) -> dict[str, Any]:
    field_tokens = {
        _field_token(key)
        for key in raw_tool
        if isinstance(key, str)
    }
    forbidden = sorted(field_tokens.intersection(_FORBIDDEN_TOOL_FIELD_TOKENS))
    if forbidden:
        raise P1025InventoryError(
            f"evaluator or task fields reached runtime discovery: {forbidden!r}"
        )
    name = raw_tool.get("name")
    if not isinstance(name, str) or not name:
        raise P1025InventoryError("raw tool name must be a nonempty string")
    description = raw_tool.get("description")
    if description is not None and not isinstance(description, str):
        raise P1025InventoryError("raw tool description must be a string when present")
    title = raw_tool.get("title")
    if title is not None and not isinstance(title, str):
        raise P1025InventoryError("raw tool title must be a string when present")
    for field_name in ("inputSchema", "outputSchema", "annotations", "_meta"):
        value = raw_tool.get(field_name)
        if field_name == "inputSchema" and not isinstance(value, Mapping):
            raise P1025InventoryError("raw tool inputSchema must be an object")
        if field_name != "inputSchema" and value is not None and not isinstance(value, Mapping):
            raise P1025InventoryError(f"raw tool {field_name} must be an object when present")
    server = raw_tool.get("server")
    if server is not None and (not isinstance(server, str) or not server):
        raise P1025InventoryError("raw tool server must be a nonempty string when present")
    return json.loads(canonical_json(dict(raw_tool)))


def _resolve_server_id(tool: Mapping[str, Any], server_ids: Sequence[str]) -> str:
    explicit_server = tool.get("server")
    if explicit_server is not None:
        if explicit_server not in server_ids:
            raise P1025InventoryError(
                f"raw tool server {explicit_server!r} is not configured"
            )
        return explicit_server

    name = tool["name"]
    matches = [server_id for server_id in server_ids if name.startswith(f"{server_id}_")]
    if len(matches) != 1:
        raise P1025InventoryError(
            f"raw tool {name!r} has no unambiguous configured server namespace"
        )
    return matches[0]


def _validated_server_ids(server_ids: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(server_ids, Sequence) or isinstance(server_ids, (str, bytes)):
        raise P1025InventoryError("configured_server_ids must be a sequence")
    normalized = tuple(sorted(_require_nonempty_string(item, "configured server ID") for item in server_ids))
    if not normalized:
        raise P1025InventoryError("at least one configured server is required")
    if len(set(normalized)) != len(normalized):
        raise P1025InventoryError("configured server IDs must be unique")
    return normalized


def _normalise_server_status(server_status: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(server_status, Mapping):
        raise P1025InventoryError("server_status must be a mapping")
    normalized: dict[str, str] = {}
    for server_id, status in server_status.items():
        normalized[_require_nonempty_string(server_id, "server status ID")] = _require_nonempty_string(
            status, "server status"
        )
    return normalized


def _require_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise P1025InventoryError(f"{field_name} must be a nonempty string")
    return value


def _field_token(field_name: str) -> str:
    """Make snake_case and camelCase leakage keys compare identically."""

    return "".join(character for character in field_name.lower() if character.isalnum())
