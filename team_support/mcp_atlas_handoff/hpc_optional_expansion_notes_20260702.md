# MCP-Atlas Optional Expansion Notes - 2026-07-02

Scope: optional follow-ups after the completed one-task HPC smoke. This remains team-support infrastructure only.

## Completed Optional Work

| optional item | status | note |
|---|---:|---|
| Slurm packaging | DONE | `run_hpc_score_drop_smoke.slurm` packages the verified baseline `1.0` versus no-code-executor `0.0` smoke |
| More perturbation exploration | DONE for one task | Filesystem-reader removals stayed `1.0`; no-code-executor removal dropped to `0.0` |
| More-than-one-task execution | NOT RUN | Requires separate approval before sending additional prompts/tool outputs to the configured LLM endpoint |

## Why More Tasks Were Not Launched

The previous approval covered the one MCP-Atlas sample task used in the smoke. Running more tasks would send additional task prompts, generated tool outputs, and evaluator payloads to the configured LLM endpoint.

Do not launch a multi-task run until the team explicitly approves that broader data send.

## Multi-Task Expansion Conditions

Before scaling beyond the one-task smoke:

1. Select compatible tasks deliberately; do not blindly use all of `sample_tasks.csv`.
2. Decide which MCP servers must be enabled for those tasks.
3. Confirm those servers start on the target host.
4. Use `--concurrency 1` for the first multi-task batch on `c2`.
5. Store raw CSVs privately.
6. Publish only summary fields unless raw outputs are reviewed.

## Practical Next Batch Shape

Recommended first expansion after approval:

| setting | value |
|---|---|
| task count | 3-5 selected tasks |
| completion model | `openai/alias-code` |
| evaluator model | `openai/alias-ha` |
| concurrency | `1` |
| baseline | original `ENABLED_TOOLS` |
| perturbation | remove one deliberately chosen required tool per task |
| report | task id, removed tool, baseline score, perturbed score, errors/timeouts |

## Current Safe Rerun Command

For the already-approved one-task smoke shape, use:

```bash
sbatch /home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/run_hpc_score_drop_smoke.slurm
```

The script itself still sends task/evaluator payloads to the configured LLM endpoint, so treat every launch as requiring normal team approval.
