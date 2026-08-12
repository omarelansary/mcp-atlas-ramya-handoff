# Adapter State

Continuously tracked state for this repository's thesis-facing role. This is
the local counterpart of `thesis-main/CURRENT_TASK.md` and
`thesis-main/THESIS_STATE.md`; it exists so the adapter's delivery position is
readable without reconstructing it from Git history.

**State date:** 2026-08-12
**Branch:** `main`
**Pin at this state:** `6039753` (2026-08-09, merge of
`p1-025-mcp-atlas-runtime-recovery`), working tree clean.

Authoritative cross-repository state remains
`thesis-main/docs/traceability/cross_repository_delivery_status.md`. Where this
file and that one disagree, that one is current and this one is stale.

## Role

| This repository owns | This repository must not own |
|---|---|
| MCP-Atlas object conversion, the dynamic MCP exposure route, the host loop, and official evaluator glue. | Reusable policy logic, general thesis architecture, or final cross-source claims. |

Reusable selection contracts and policy logic live in the pinned
`active-registry-core` package in `thesis-main`. This fork may convert
MCP-Atlas objects, execute the host loop, and invoke the MCP-Atlas evaluator,
and nothing more. See `AGENTS.md` for the full interface boundary and the
selector input restrictions.

## Delivered and stable

- **P1-016** dynamic MCP-Atlas evaluation loop: a separate
  `/v2/mcp_eval/run_agent_dynamic` route that rediscovers raw tools each
  completion cycle, exposes only the host-selected active set, and blocks
  hidden-tool calls. The official fixed-list `/v2/mcp_eval/run_agent` path and
  its `enabledTools` schema are unchanged.
- **P1-017** dynamic selectors and runs: registered dynamic full-exposure,
  stateless-semantic, and called-tool-retention selectors, plus the development
  grid and the scored held-out run. Schema-byte reductions of about 82-83% were
  measured against dynamic full exposure, with low absolute official coverage in
  every condition — an adapter pilot observation, not a policy winner.
- **P1-018** typed, prompt-free failure telemetry for the dynamic route.
- **P1-025** runtime recovery: the explicitly modified container/dependency
  build that reaches and holds the declared `20/20/0` no-key server profile and
  produces matching canonical `126`-tool inventory hashes, plus the
  offline-tested no-provider inventory-capture utility. This is an explicitly
  modified runtime, never the unmodified official image, and every downstream
  record carries that disclosure.

## Current position

**No adapter task is open.** The P1-025 tool-selection-policy line closed on
2026-08-12 as a documented negative result, and its diagnostic-layer work
(thirteen non-learned selection methods across four pre-registered batches) ran
entirely in `thesis-main` against local captures — no change to this repository
was required or made for any of it.

This repository was load-bearing for that line in two places only, both
unchanged since: it supplied the qualified runtime and inventory capture, and
its `services/mcp_eval` completion and evaluator services ran the
evaluator-repeatability check and the executable-layer comparison (639 live
completions and 639 live evaluations over the 71 development rows, in which
full exposure clearly outperformed both reduced-tool conditions).

## What could reopen work here

One thing, and it is not yet decided: the recorded next action for P1-025 is a
D005 scope decision about whether the active registry may extend its active set
later in the same session, instead of selecting once per task with no recovery
step. That decision is confined to component 4's action space and explicitly
does **not** open component 5 execute/expand/defer/abstain routing or
abstention.

If it relaxes the boundary toward session-scoped selection, this repository
becomes relevant again — but as existing precedent rather than new
infrastructure. The dynamic route already re-selects per completion cycle, and
the P1-017 `called-tool-retention` selector is already session-stateful.
Nothing should be built here in anticipation of that decision; wait for the
recorded decision first, then register a scoped adapter task.

## Update rule

Update this file in the same coherent delivery when any of these change:

- this repository's stable `main` pin or branch position;
- a completed adapter checkpoint or its supported/unsupported claim;
- the interface boundary between this fork and `active-registry-core`; or
- the recorded next adapter action.

Do not record raw prompts, trajectories, GTFA claims, tool results, model
responses, evaluator payloads, or credentials here. Link to the private record
instead.
