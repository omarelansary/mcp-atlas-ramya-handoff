# MCP-Atlas Ramya Handoff Packet

Purpose: make the MCP-Atlas setup usable for Ramya/team capability-gap data generation from the shared `mcp-atlas` repo, without requiring access to Omar's private thesis folder.

## Start Here

Read these first:

1. `ramya_start_here_20260707.md`
   - First-read handoff note explaining what is set up, what the smoke test proves, and how to interpret tool-removal cases.

2. `ramya_delivery_note.md`
   - What to send, what not to send, and how Ramya should use the packet.

3. `ramya_mcp_atlas_runbook.md`
   - The practical team-facing runbook.
   - Uses the verified baseline `1.0` versus no-code-executor `0.0` smoke.

4. `ramya_capability_gap_dataset_support_20260707.md`
   - Explains why MCP-Atlas can generate candidate gaps but does not provide final capability-gap labels by itself.

5. `hpc_phase_closeout_20260702.md`
   - Short proof that the original prompt goals and extended phases are closed.

## Files

- `ramya_delivery_note.md`: plain-language delivery note.
- `ramya_start_here_20260707.md`: first-read handoff note.
- `ramya_mcp_atlas_runbook.md`: HPC-verified team runbook.
- `ramya_capability_gap_dataset_support_20260707.md`: capability-gap dataset interpretation note.
- `hpc_phase_closeout_20260702.md`: closure checklist.
- `hpc_mcp_atlas_smoke_status_20260702.md`: detailed evidence, command pattern, output filenames, blockers, and fixes.
- `hpc_optional_expansion_notes_20260702.md`: notes for future multi-task expansion and approval boundaries.
- `run_hpc_score_drop_smoke.slurm`: Slurm-ready rerun script for the baseline `1.0` versus no-code-executor `0.0` smoke.
- `mcp_eval.env.example`: template for the ignored local secret file.
- `boundary_and_support_note.md`: explains that this is team-support infrastructure that also de-risks Omar's later thesis runs.

## Required Local Secret File

Create this file on the host before running the Slurm script:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/.env
```

Use `mcp_eval.env.example` as the template. Do not commit the real `.env`; it is ignored by Git.

## Verified Result

| run | removed tool(s) | coverage score |
|---|---|---:|
| baseline | none | `1.0` |
| no-code-executor perturbation | `mcp-code-executor_execute_code` | `0.0` |

This is a team-support infrastructure smoke and candidate score-drop example, not a final capability-gap label and not a thesis claim.

## Rerun

After approval to send the selected task prompt, generated tool outputs, and evaluator payloads to the configured LLM endpoint:

```bash
sbatch /home/omel305g/masters_workspace/mcp-atlas/team_support/mcp_atlas_handoff/run_hpc_score_drop_smoke.slurm
```

The script writes a safe summary under:

```text
/tmp/mcpatlas_eval_tmp/logs/<run-label>/summary.tsv
```

Reruns create raw completion and evaluator CSVs under:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/
```

Those raw CSVs may contain prompts, trajectories, conversation history, and GTFA claims. Keep them private unless reviewed. The verified 2026-07-02 raw outputs are not included in this shared handoff packet.
