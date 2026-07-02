# Ramya MCP-Atlas Runbook

Status: HPC-verified team-support runbook based on the 2026-07-02 smoke on host `c2`.

This is a team-support infrastructure for producing MCP-Atlas capability-gap examples with the official completion and evaluator pipeline.

## Verified Smoke Result

Reference task:

```text
689bd255c0422b257e7dfcc5
```

Models:

- Completion: `openai/alias-code`
- Evaluator: `openai/alias-ha`

Scores on host `c2`:

| run | removed tool(s) | score |
|---|---|---:|
| baseline | none | `1.0` |
| no-code-executor perturbation | `mcp-code-executor_execute_code` | `0.0` |

This is the concise one-task capability-gap smoke result to use for handoff. Other filesystem-reader perturbations were also tested, but they still scored `1.0` because substitute filesystem tools remained available.

This supports Omar's later thesis runs by verifying the official MCP-Atlas execution path, but it is not Omar's thesis method. See `boundary_and_support_note.md`.

## Paths

Repository:

```text
/home/omel305g/masters_workspace/mcp-atlas
```

Protected env file:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/.env
```

Use `mcp_eval.env.example` in this handoff folder as the template. Do not commit the real `.env`; it is ignored by Git. The verified commands load it with `python-dotenv`.

Inputs:

```text
/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/baseline_689bd255c0422b257e7dfcc5.csv
/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/perturbed_no_code_executor_689bd255c0422b257e7dfcc5.csv
```

Outputs:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/
```

Full command details and exact output filenames:

```text
/home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/hpc_mcp_atlas_smoke_status_20260702.md
```

Slurm-ready rerun script:

```text
/home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/run_hpc_score_drop_smoke.slurm
```

## Runtime Pattern

Docker is not available on `c2`. Use the extracted Apptainer image rootfs plus `bwrap` workaround already captured in the smoke status.

Start these two services:

1. Agent service on `127.0.0.1:1984`
2. `mcp_eval` service on `127.0.0.1:3000`

Health checks:

```bash
curl -s http://127.0.0.1:1984/enabled-servers
curl -s http://127.0.0.1:3000/health
```

Expected status:

```text
filesystem OK
mcp-code-executor OK
mcp_eval healthy
```

## Run Baseline

Run from:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval
```

Inside the verified `bwrap`/`dotenv` command pattern, use:

```bash
uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py \
  --model openai/alias-code \
  --input /tmp/mcpatlas_inputs/baseline_689bd255c0422b257e7dfcc5.csv \
  --output hpc_smoke_baseline_alias_code_20260702.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter
```

Then score:

```bash
uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py \
  --input-file completion_results/hpc_smoke_baseline_alias_code_20260702.csv \
  --model-label hpc_smoke_baseline_alias_code_20260702 \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1
```

Expected score: `1.0`.

## Run Capability-Gap Perturbation

Use the no-code-executor input:

```text
/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/perturbed_no_code_executor_689bd255c0422b257e7dfcc5.csv
```

This keeps the task, prompt, trajectory, and GTFA claims unchanged. It changes only `ENABLED_TOOLS` by removing:

```text
mcp-code-executor_execute_code
```

Inside the verified `bwrap`/`dotenv` command pattern, use:

```bash
uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py \
  --model openai/alias-code \
  --input /tmp/mcpatlas_inputs/perturbed_no_code_executor_689bd255c0422b257e7dfcc5.csv \
  --output hpc_smoke_no_code_executor_alias_code_20260702.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter
```

Then score:

```bash
uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py \
  --input-file completion_results/hpc_smoke_no_code_executor_alias_code_20260702.csv \
  --model-label hpc_smoke_no_code_executor_alias_code_20260702 \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1
```

Expected score: `0.0`.

## Minimal Team Report

Report only safe summary fields unless raw outputs have been reviewed:

| field | value |
|---|---|
| task id | `689bd255c0422b257e7dfcc5` |
| baseline score | `1.0` |
| perturbation | remove `mcp-code-executor_execute_code` |
| perturbed score | `0.0` |
| completion model | `openai/alias-code` |
| evaluator model | `openai/alias-ha` |
| host | `c2` |
| status | official pipeline ran end to end |

## Batch Rerun

The smoke can be rerun through Slurm with:

```bash
sbatch /home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/run_hpc_score_drop_smoke.slurm
```

The script:

- prepares the baseline and no-code-executor one-row inputs from `sample_tasks.csv`;
- starts the agent service and `mcp_eval`;
- runs completion and evaluator scoring for both rows;
- writes a safe summary under `/tmp/mcpatlas_eval_tmp/logs/<run-label>/summary.tsv`;
- stops the services at job exit.

Use this only after approval to send the selected task prompt, generated tool outputs, and evaluator payloads to the configured LLM endpoint.

## Safety

- Do not publish raw CSVs without review; they include prompts, trajectories, conversation history, and GTFA claims.
- Do not store API keys in Git.
- Do not print secret values in logs or notes.
- Treat this as a one-task infrastructure smoke, not a scientific result.
- Keep this team-support workflow separate from Omar's thesis active-registry method.
