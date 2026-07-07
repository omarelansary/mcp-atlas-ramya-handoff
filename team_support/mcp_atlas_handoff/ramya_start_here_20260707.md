# Ramya MCP-Atlas Handoff - Start Here

Date: 2026-07-07

Status: team-support handoff note. This is not Omar's thesis method and not a final capability-gap dataset contract.

## What Is Already Set Up

This repository is a shared MCP-Atlas handoff repo:

```text
https://github.com/omarelansary/mcp-atlas-ramya-handoff
```

It contains the MCP-Atlas code plus this handoff packet:

```text
team_support/mcp_atlas_handoff/
```

The handoff packet contains:

- a verified one-task HPC smoke result;
- a Slurm rerun script;
- a safe `.env` template;
- notes on what can and cannot be inferred from tool-removal runs.

## What This Setup Is For

This setup is for generating and checking candidate capability-gap examples with MCP-Atlas.

It can support this workflow:

```text
original MCP-Atlas task
  -> run with original ENABLED_TOOLS
  -> run again with a selected tool/server hidden
  -> compare final-answer coverage
  -> inspect whether the missing tool is genuinely non-substitutable
```

It is not a finished dataset by itself.

## What The Existing Smoke Test Proves

The one-task smoke proved the infrastructure path:

```text
MCP-Atlas agent environment works.
MCP-Atlas mcp_eval service works.
The configured LLM endpoint can call tools.
The official evaluator can score the final answer.
Changing ENABLED_TOOLS changes what the model can see.
```

It also produced one candidate score-drop example:

| task | condition | score |
|---|---|---:|
| `689bd255c0422b257e7dfcc5` | original tools | 1.0 |
| `689bd255c0422b257e7dfcc5` | remove `mcp-code-executor_execute_code` | 0.0 |

Important boundary:

```text
The ablation input was generated outside native MCP-Atlas dataset labels.
It was then executed and scored through the official MCP-Atlas harness.
```

So this is a pipeline/intervention smoke and a candidate example, not a completed capability-gap dataset.

## What Ramya Should Read

Read in this order:

1. `ramya_mcp_atlas_runbook.md`
   - How the verified HPC smoke is run and scored.

2. `ramya_capability_gap_dataset_support_20260707.md`
   - Why MCP-Atlas is useful but does not already provide final capability-gap labels.

3. `hpc_mcp_atlas_smoke_status_20260702.md`
   - Detailed technical evidence for the successful HPC run.

4. `run_hpc_score_drop_smoke.slurm`
   - Batch script to rerun the one-task smoke on HPC.

## Capability-Gap Labeling Rule

Do not label every removed-tool run as a capability gap.

Use three levels:

```text
tool removal:
  a tool/server was hidden from ENABLED_TOOLS.

stable ablation failure:
  original condition succeeds, hidden-tool condition fails across repeated runs.

validated capability gap:
  stable ablation failure plus evidence that the removed capability is not reasonably replaceable by model knowledge, simple reasoning, or another visible tool.
```

Examples of stronger non-substitutable capabilities:

- local or private files;
- private datasets;
- live API data;
- database access;
- code execution;
- authenticated workspace access.

Examples of weak candidates:

- calculator removal for simple arithmetic;
- common-knowledge lookup;
- simple text transformation;
- cases where another visible tool can do the same work.

## Recommended First Dataset Procedure

Use a small, controlled procedure before scaling:

1. Select tasks where the original MCP-Atlas run succeeds.
2. Choose a required tool/server from the reference trajectory.
3. Remove or hide that tool/server in `ENABLED_TOOLS`.
4. Repeat the original and ablated runs, ideally 3-5 times each if token budget allows.
5. Mark `stable_ablation_failure` only if original succeeds reliably and ablated fails reliably.
6. Promote to `validated_capability_gap` only if the removed capability is non-substitutable.

This keeps the dataset from confusing ordinary model failure with missing-capability failure.

## What To Say In Plain Terms

```text
The MCP-Atlas repo/harness is set up as team-support infrastructure. The previous smoke test mainly verified that MCP servers, tool calling, output scoring, and ENABLED_TOOLS perturbation work end to end.

The ablation was generated outside native MCP-Atlas labels, then executed through the official MCP-Atlas harness. It is not a built-in MCP-Atlas capability-gap label.

For the dataset, use this setup to generate candidate gaps, but keep three labels separate: tool removal, stable ablation failure, and validated capability gap.
```

## Safety

- Do not commit or share `.env`.
- Do not publish raw completion/evaluation CSVs without review.
- Raw outputs may include prompts, trajectories, tool outputs, and `GTFA_CLAIMS`.
- Keep this team-support workflow separate from Omar's thesis evidence unless explicitly reviewed.
