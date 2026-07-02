# MCP-Atlas HPC Phase Closeout - 2026-07-02

Scope: closure checklist for the HPC/server prompt and the follow-up perturbation phases. This is team-support infrastructure only.

## Original Prompt Goals

| goal | status | evidence |
|---|---:|---|
| Check whether the environment can run the official agent environment | DONE | Agent ran on `127.0.0.1:1984`; `filesystem` and `mcp-code-executor` online |
| Check whether the environment can run `mcp_eval` | DONE | `mcp_eval` ran on `127.0.0.1:3000`; `/health` returned healthy |
| Run one baseline task using original `ENABLED_TOOLS` | DONE | `hpc_smoke_baseline_alias_code_20260702.csv` |
| Run one perturbed task with tools removed | DONE | Filesystem-reader perturbations and no-code-executor perturbation were run |
| Run the official evaluator on both outputs | DONE | Scored CSVs and coverage stats were produced |
| Produce a short team-facing usage guide | DONE | `ramya_mcp_atlas_runbook.md` updated with HPC-verified result |

## Phase Status

| phase | status | result |
|---|---:|---|
| Phase 0: recovery audit | DONE | Existing setup and missing outputs identified |
| Phase 1: restart local services | DONE | Agent and `mcp_eval` verified |
| Phase 2: approval and secret gate | DONE | Explicit approval received; secrets loaded through protected `.env` |
| Phase 3: baseline completion | DONE | Baseline completion produced one row |
| Phase 4: first perturbation completion | DONE | Filesystem-reader perturbation produced one row |
| Phase 5: official evaluator scoring | DONE | Baseline and perturbation scored |
| Phase 6: team-facing handoff update | DONE | Smoke status and runbook updated |
| Extended: stronger filesystem-reader perturbation | DONE | Score stayed `1.0` |
| Extended: no-code-executor perturbation | DONE | Score dropped to `0.0` |
| Optional: Slurm packaging | DONE | `run_hpc_score_drop_smoke.slurm` added and `bash -n` validated |
| Optional: multi-task expansion note | DONE | `hpc_optional_expansion_notes_20260702.md` added; additional LLM sends require separate approval |

## Final Smoke Result

| run | removed tool(s) | coverage score |
|---|---|---:|
| baseline | none | `1.0` |
| no-code-executor perturbation | `mcp-code-executor_execute_code` | `0.0` |

This closes the deployment checkpoint and gives the team a concise one-task capability-gap smoke.

## Canonical Handoff Files

- `ramya_mcp_atlas_runbook.md`: shortest team-facing run instruction.
- `hpc_mcp_atlas_smoke_status_20260702.md`: detailed evidence, command pattern, output filenames, blockers/fixes.
- `run_hpc_score_drop_smoke.slurm`: Slurm-ready rerun script for the verified one-task smoke.

## Not Required For Closure

These are optional follow-ups, not blockers:

- Run more than one task.
- Explore additional perturbation strategies.
- Move the workflow to a Docker-capable VM.

The Slurm batch script has now been added. Additional task runs would send additional prompts/tool outputs to the configured LLM endpoint and should be approved separately before execution.

## Safety Boundary

Raw completion and evaluator CSVs may contain prompts, trajectories, conversation history, and GTFA claims. Keep them private unless reviewed.
