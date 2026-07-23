# P1-017 Frozen Cohort Grid

This runner performs only predeclared dynamic MCP-Atlas cohort grids. It never
accepts a client-supplied active-tool list:

- `--cohort development` is T2: five development tasks, three registered
  selectors, budgets `10` and `20`, and two repetitions (`60` requests).
- `--cohort heldout` is T3: five held-out tasks, the T2-selected budget `10`,
  the same selectors, and two repetitions (`30` requests).

The held-out cohort is hard-coded and can be executed only after the tracked
T2 decision has selected its budget.

## Dry Preflight

Run from `services/mcp_eval`:

```powershell
python scripts/run_p1_017_dynamic_development_grid.py `
  --input sample_tasks.csv `
  --model openai/alias-code `
  --run-id p1_017_t2_dryrun
```

Without `--execute`, the command only verifies the frozen source SHA-256 and
prints the planned grid. It does not contact an MCP server, completion model,
or evaluator.

The held-out dry preflight is:

```powershell
python scripts/run_p1_017_dynamic_development_grid.py `
  --cohort heldout `
  --input sample_tasks.csv `
  --model openai/alias-code `
  --run-id p1_017_t3_dryrun
```

## Execution Gate

`--execute` additionally requires `--execution-manifest <path>`. The manifest
is a credential-free JSON object with these fields:

- `source_pin` and `source_task_file_sha256`
- `core_release_tag` and `core_commit`
- `adapter_commit`
- `container_image` and `server_mode`
- `completion_model`, `evaluator_model`, and `evaluator_pass_threshold` (`0.75`)
- `max_turns`
- `provider_parameters`

The runner verifies the source pin/hash, completion model, and maximum turns.
It rejects keys that look like credentials. Keep API keys only in `.env`.

Completion rows and the safe structural manifest are written under ignored
`completion_results/p1_017_t2/<run-id>/` or
`completion_results/p1_017_t3/<run-id>/`. The evaluator is intentionally a
separate `mcp_evals_scores.py` invocation. A dynamic endpoint HTTP 500 caused
by a model request for a tool outside the visible set is recorded as
`hidden_tool_request` and the grid continues; connection and timeout failures
stop the run.
