# Ramya Delivery Note - MCP-Atlas HPC Smoke

Purpose: explain exactly what Omar should hand to Ramya/team and how they should use it.

## What To Deliver

Give Ramya access to:

```text
/home/omel305g/masters_workspace/mcp-atlas/
```

Point her first to:

```text
/home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/README.md
```

The main files in that packet are:

1. `ramya_mcp_atlas_runbook.md`
   - Main user-facing instructions.
   - Shows the verified baseline `1.0` versus no-code-executor `0.0` smoke.

2. `hpc_phase_closeout_20260702.md`
   - Short checklist proving the original prompt goals are done.
   - Good for a quick status update.

3. `run_hpc_score_drop_smoke.slurm`
   - Batch script to rerun the same one-task smoke on the HPC/server.
   - It reruns baseline and no-code-executor perturbation.

4. `boundary_and_support_note.md`
   - Explains that this is team-support infrastructure that also de-risks Omar's later thesis runs.

## What Not To Deliver Publicly

Do not send or publish these unless reviewed:

- API keys or `.env` files.
- Raw completion CSVs.
- Raw evaluator CSVs.
- Raw prompts, trajectories, conversation histories, or GTFA claims.

Reruns create raw artifacts under:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/
```

The verified 2026-07-02 raw outputs are not included in the shared handoff packet.

## What To Tell Ramya

Short version:

```text
MCP-Atlas was verified on host c2 using the official pipeline. The baseline task scored 1.0. Removing mcp-code-executor_execute_code made the same task score 0.0. The runbook and Slurm rerun script are in team_support/mcp_atlas_handoff/.
```

Use this result as a team-support smoke, not as a thesis claim.

## How Ramya Uses It

If she only needs the result, read:

```text
ramya_mcp_atlas_runbook.md
hpc_phase_closeout_20260702.md
```

If she wants to rerun the smoke, first confirm approval to send the one sample task prompt, generated tool outputs, and evaluator payloads to the configured LLM endpoint.

Then run:

```bash
sbatch /home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/run_hpc_score_drop_smoke.slurm
```

Expected safe summary:

```text
baseline              none                           1.0
no-code-executor      mcp-code-executor_execute_code 0.0
```

If she wants more than one task, use `hpc_optional_expansion_notes_20260702.md` first. Do not blindly run all of `sample_tasks.csv`.
