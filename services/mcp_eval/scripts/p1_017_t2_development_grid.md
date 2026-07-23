# P1-017 T2 Development Grid

This runner performs the predeclared dynamic MCP-Atlas development grid only:
five development tasks, three registered selector conditions, budgets `10` and
`20`, and two repetitions (`60` completion requests). It never accepts a
client-supplied active-tool list.

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
`completion_results/p1_017_t2/<run-id>/`. The evaluator is intentionally a
separate `mcp_evals_scores.py` invocation: freeze its model and configuration
before scoring, and do not run held-out tasks until the development rule picks
one common budget.
