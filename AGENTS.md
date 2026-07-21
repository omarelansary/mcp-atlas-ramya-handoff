# MCP-Atlas Fork Agent Instructions

This fork has two separate roles: preserve the upstream MCP-Atlas evaluation
path and host the thesis-specific dynamic MCP exposure path. Treat them as
separate interfaces.

## Before Editing

1. Inspect `git status -sb`, the current branch, its merge base, and its diff
   from `origin/main`.
2. Read the current P1 task/state in `thesis-main` and the private P1-017
   registry before changing dynamic-selector behavior.
3. Use one branch for one task. Start P1-017 work from current `origin/main`,
   not from an archival or handoff branch.

## Interface Boundary

- Keep the official fixed-list `/v2/mcp_eval/run_agent` path behavior intact.
- Put thesis dynamic exposure only behind the separate dynamic route and
  request model. Do not silently change the official route to use a selector.
- A selector may use only the frozen permitted inputs: raw MCP tool fields,
  model-visible messages, prior tool-call names, and state derived from them.
  It must not consume evaluator labels, trajectories, GTFA claims, tool
  results, or post-hoc required-tool information.
- Preserve raw MCP tool order for full exposure and enforce hidden-tool
  rejection before forwarding a tool call.

## Data And Git Safety

- Never commit `.env` files, keys, prompts, trajectories, GTFA claims, raw tool
  results, model responses, evaluator payloads, completion CSVs, or evaluation
  CSVs. The result directories are intentionally ignored.
- Stage explicit paths only. Do not use `git add -A` in a mixed worktree.
- Do not rebase, reset, force-push, delete branches, drop stashes, or discard
  existing work without explicit user approval and a preservation record.

## Validation And Publishing

- Add fake-provider/fake-MCP tests before a completion provider or evaluator is
  called. Tests must demonstrate selector visibility, ordering, forwarding,
  hidden-call rejection, and evaluator-field isolation.
- Run the relevant deterministic unit tests and `git diff --check` before
  committing. Record blocked environment checks rather than assuming success.
- Keep a focused commit history and merge only a branch whose scope matches its
  name and task record.
