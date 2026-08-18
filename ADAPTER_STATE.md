# Adapter State

Continuously tracked state for this repository's thesis-facing role. This is
the local counterpart of `thesis-main/CURRENT_TASK.md` and
`thesis-main/THESIS_STATE.md`; it exists so the adapter's delivery position is
readable without reconstructing it from Git history.

**State date:** 2026-08-17
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

## 2026-08-17 — five harness changes made for P1-025-V1, all uncommitted

V1 (MCP-Atlas harness validation) executed against this adapter, driving each
row's own `ENABLED_TOOLS` through `/v2/mcp_eval/run_agent`. It passed at
`0.5730` against a `0.685` reference. Five changes to this repository were
required, four of them defect fixes. **All are currently uncommitted.**

### 1. Transient tool timeouts are retried — `mcp_client/sandbox_client.py`

Three attempts with an escalating wait, `60s → 120s → 180s`, configurable via
`TOOL_CALL_ATTEMPTS` (set `1` to restore the previous behaviour). Only
timeouts retry; a non-200 tool response is still returned to the model as an
error result, and other exceptions still fail immediately.

Placed in the client rather than the agent loop deliberately: the `MCPClient`
interface is unchanged, every test fake still satisfies it, and **the dynamic
route inherits the same protection**, which matters because P1-026 scores
failure as a policy outcome.

Measured effect: the V1 probe lost 1 row of 5 to a 60s `wikipedia_get_summary`
timeout; the full 89-row run had **zero** infrastructure failures.

Disclosed limit: upstream states it added transient-error retry but publishes
neither attempt count nor backoff, so these values are ours. This narrows the
gap to upstream; it does not close it.

### 2. Token accounting is surfaced — `llm.py`, `agent_eval.py`

`LLMResponse` now carries `usage`, including `reasoning_tokens` and
`cached_prompt_tokens` where the provider reports them. `run_mcp_eval`
accumulates it and emits one `AgentOutput("usage", …)` at the end of the run,
carrying `transient_tool_retries` alongside.

Additive by design: every existing consumer filters on `type == "message"`, so
nothing else changes. This is what let V1 report measured rather than projected
cost.

### 3. Silent-scoring defect fixed — `mcp_completion_script.py`

The model-response extraction had its role checks at the **outer** indent
level, so a stream whose last entry was not a `"message"` read `msg` before it
was bound. The `UnboundLocalError` was swallowed by a bare
`except Exception: pass`, leaving `script_model_response` as `None`: the row
scored as a failure with no sign anything had gone wrong.

Verified **dormant** before the fix — the newly added `"usage"` output is the
only non-`message` type in the codebase, so **no existing result is affected** —
but it would have fired on every V1 row.

### 4. Judge routing fixed for non-Gemini providers — `mcp_evals_scores.py`

Three defects, all of which produced *believable wrong numbers* rather than
errors, because the evaluator scores a failed judge call as `0`:

- `EVAL_LLM_BASE_URL` was applied to every model, sending `gemini/*` traffic to
  an OpenAI-compatible gateway. `resolve_api_base` now drops a gateway URL for
  natively-routed providers.
- Gemini's `response_format.response_schema` was sent to every provider; real
  OpenAI rejects it. Now branched per provider, with the schema delivered in
  the **prompt** for OpenAI, which also satisfies its requirement that the
  messages contain the literal word "json".
- The `gpt-5` temperature special case matched only the literal string
  `"gpt-5"`, so `gpt-5.4` would have been sent an unsupported `temperature=0.0`.

Measured on identical rows: schema rejected → coverage `0.000`; schema fixed
but no "json" in prompt → `0.500`; both fixed → `0.917`.

**The ScaDS OpenAI-compatible path is deliberately byte-identical to before**,
so Phase 2b and all existing scored numbers remain comparable.

### 5. Off-contract judge outcomes are normalised and counted — `mcp_evals_scores.py`

Gemini enforces the `coverage_outcome` enum through `response_schema`; an
OpenAI-compatible gateway does not. Observed on ScaDS: Kimi-K3 returns
`unable_to_determine`, GLM-5.2 returns `unfulfilled`. The previous
`.get(outcome, 0.0)` default scored every such claim `0` silently, so a judge
that merely words things differently was indistinguishable from a model that
failed.

Synonyms are now normalised, and anything still unrecognised is **counted and
logged** rather than quietly zeroed.

**Open follow-up:** the correct fix is to make the provider *enforce* the
schema rather than to clean up afterwards — ScaDS ignores Gemini's
`response_schema` but may honour OpenAI's `json_schema` strict mode or vLLM's
`guided_json`. Untested at the time of writing; the normaliser is a backstop,
not the answer.

### Secret hygiene

`.gitignore` had `.env` only, which does not match `.env.bak-*`; a credential
backup was therefore trackable. Now `.env.*` with `.env.example` and
`.env.template` excepted.

### Test status

`61 passed, 7 subtests` after every change above.

## Update rule

Update this file in the same coherent delivery when any of these change:

- this repository's stable `main` pin or branch position;
- a completed adapter checkpoint or its supported/unsupported claim;
- the interface boundary between this fork and `active-registry-core`; or
- the recorded next adapter action.

Do not record raw prompts, trajectories, GTFA claims, tool results, model
responses, evaluator payloads, or credentials here. Link to the private record
instead.
