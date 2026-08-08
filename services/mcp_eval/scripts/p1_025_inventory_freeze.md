# P1-025 Runtime Inventory Freeze

This script captures the MCP-Atlas default no-key tool inventory before any
P1-025 policy or task evaluation. It does not read dataset rows, call a tool,
call an LLM, or invoke the evaluator.

Start the pinned default no-key agent environment first. Then run from
`services/mcp_eval`:

```bash
python scripts/run_p1_025_inventory_freeze.py \
  --raw-output-dir completion_results/p1_025/inventory_freeze/raw \
  --safe-manifest completion_results/p1_025/inventory_freeze/p1_025_r1_safe_manifest.json
```

The runner accepts raw output only below `completion_results/` or
`evaluation_results/`, which are ignored by Git. Do not commit its tool
records. The safe manifest contains only pins, aggregate counts, anonymous
tools-per-server statistics, hashes, status counts, and scope flags. It does
not contain raw tool names, schemas, descriptions, task rows, trajectories,
claims, tool outputs, or model output.

MCP-Atlas's flattened raw tool records do not include a server field in the
preserved source snapshot. The capture utility therefore accepts an origin only
when the tool has an explicit configured `server` field or its name has exactly
one configured `<server-id>_` namespace prefix. It fails instead of guessing.

The command fails when fewer than 20 configured default servers are online,
any tool lacks a unique configured server namespace, or the two discovery
snapshots differ. A successful manifest is a runtime gate, not a policy result
or a frozen task cohort. The next step is private cohort construction from all
eligible public rows under this exact manifest.
