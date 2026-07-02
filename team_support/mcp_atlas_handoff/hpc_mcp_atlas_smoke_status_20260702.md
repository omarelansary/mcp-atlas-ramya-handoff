# MCP-Atlas HPC Smoke Status - 2026-07-02

Scope: team-support infrastructure smoke only. This does not validate thesis claims.

## Approval

User explicitly approved sending the one MCP-Atlas sample task prompt and generated tool outputs to the configured LLM endpoint for:

- baseline completion;
- perturbed completion;
- evaluator scoring for both outputs.

No API keys or raw task rows are included in this note.

## Environment Status

| Component | Status | Evidence |
|---|---:|---|
| Host | PASS | `c2` |
| Main checkout | PASS | `/home/omel305g/masters_workspace/mcp-atlas` at `0f307af813334c5174dc0b560c29ce3d5828ee50` |
| Docker | FAIL | Not available on path |
| Apptainer/Singularity | PASS with workaround | Extracted image rootfs plus `bwrap` path works |
| Agent service | PASS | `GET /enabled-servers` returned `filesystem` and `mcp-code-executor` online |
| Agent tools | PASS | `POST /list-tools` returned 23 tools |
| `mcp_eval` service | PASS | `/docs` returned HTTP 200; `/health` returned `{"status":"healthy"}` |
| LLM endpoint | PASS for this smoke | Completion and evaluator calls succeeded through protected `.env` |
| Baseline completion | PASS | One task completed in 8.1s |
| Perturbed completion | PASS | One task completed in 8.8s |
| Baseline evaluator | PASS | Coverage score `1.0` |
| Perturbed evaluator | PASS | Coverage score `1.0` |

## Inputs

Prepared one-row inputs:

```text
/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/baseline_689bd255c0422b257e7dfcc5.csv
/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/perturbed_689bd255c0422b257e7dfcc5.csv
```

Input metadata:

| input | rows | enabled tools | removed tools |
|---|---:|---:|---|
| baseline | 1 | 14 | none |
| perturbed | 1 | 12 | `filesystem_read_file`, `filesystem_read_text_file` |
| stronger perturbation | 1 | 11 | `filesystem_read_file`, `filesystem_read_text_file`, `filesystem_read_multiple_files` |
| no-code-executor perturbation | 1 | 13 | `mcp-code-executor_execute_code` |

## Results

| run | completion rows | score rows | coverage score | errors | trajectory time | attempts |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | `1.0` | no | `8.06026029586792` | 1 |
| perturbed | 1 | 1 | `1.0` | no | `8.840484619140625` | 1 |
| stronger perturbation | 1 | 1 | `1.0` | no | `6.097702264785767` | 1 |
| no-code-executor perturbation | 1 | 1 | `0.0` | no | `2.538119316101074` | 1 |

Observed tool names only, without raw arguments or raw trajectory content:

| run | observed tool names |
|---|---|
| baseline | `filesystem_list_allowed_directories`, `filesystem_read_text_file`, `filesystem_search_files`, `mcp-code-executor_execute_code` |
| perturbed | `filesystem_list_allowed_directories`, `filesystem_read_multiple_files`, `filesystem_search_files`, `mcp-code-executor_execute_code` |
| stronger perturbation | `filesystem_directory_tree`, `filesystem_get_file_info`, `filesystem_list_allowed_directories`, `filesystem_list_directory`, `mcp-code-executor_execute_code` |
| no-code-executor perturbation | `filesystem_list_allowed_directories`, `filesystem_list_directory`, `filesystem_read_text_file`, `mcp-code-executor_read_code_file` |

Interpretation for team-support smoke:

- The official pipeline ran end to end on host `c2`.
- The perturbation did not lower the score because `filesystem_read_multiple_files` remained available and the model used it.
- The stronger perturbation also did not lower the score because other filesystem inspection tools plus code execution remained sufficient.
- The no-code-executor perturbation produced the desired score drop: coverage score `0.0`.
- This is useful infrastructure evidence, not a thesis-method result.
- The practical one-task capability-gap example for this host is now baseline `1.0` versus no-code-executor `0.0`.

## Output Files

Completion outputs produced during verification:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/hpc_smoke_baseline_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/hpc_smoke_perturbed_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/hpc_smoke_no_fs_readers_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/completion_results/hpc_smoke_no_code_executor_alias_code_20260702.csv
```

Evaluator outputs produced during verification:

```text
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/scored_hpc_smoke_baseline_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_stats_hpc_smoke_baseline_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_histogram_hpc_smoke_baseline_alias_code_20260702.png
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/scored_hpc_smoke_perturbed_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_stats_hpc_smoke_perturbed_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_histogram_hpc_smoke_perturbed_alias_code_20260702.png
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/scored_hpc_smoke_no_fs_readers_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_stats_hpc_smoke_no_fs_readers_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_histogram_hpc_smoke_no_fs_readers_alias_code_20260702.png
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/scored_hpc_smoke_no_code_executor_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_stats_hpc_smoke_no_code_executor_alias_code_20260702.csv
/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/evaluation_results/coverage_histogram_hpc_smoke_no_code_executor_alias_code_20260702.png
```

These files may contain `PROMPT`, raw conversation history, trajectories, and GTFA claims. They are not included in the shared handoff packet; rerunning the Slurm script will create fresh raw outputs under the same MCP-Atlas result directories.

## Working Command Pattern

Use these variables:

```bash
ROOT=/tmp/mcpatlas_agent_tree
REPO=/home/omel305g/masters_workspace/mcp-atlas
SECRET_ENV=/home/omel305g/masters_workspace/mcp-atlas/services/mcp_eval/.env
EVAL_TMP=/tmp/mcpatlas_eval_tmp
UV_CACHE=/tmp/mcpatlas_uv_cache
```

Start agent service:

```bash
bwrap \
  --bind "$ROOT" / \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --chdir /agent-environment \
  --setenv ENABLED_SERVERS filesystem,mcp-code-executor \
  --setenv PATH /agent-environment/.venv/bin:/usr/local/bin:/usr/bin:/bin \
  --setenv HOME /root \
  --setenv VIRTUAL_ENV /agent-environment/.venv \
  /agent-environment/entrypoint.sh \
  /agent-environment/.venv/bin/python -m uvicorn agent_environment.main:app --host 127.0.0.1 --port 1984
```

Start `mcp_eval` service:

```bash
bwrap \
  --bind "$ROOT" / \
  --bind "$REPO" /workspace/mcp-atlas \
  --ro-bind "$SECRET_ENV" /run/secrets/mcp_atlas.env \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --proc /proc \
  --dev /dev \
  --bind "$EVAL_TMP" /tmp \
  --bind "$UV_CACHE" /root/.cache/uv \
  --chdir /workspace/mcp-atlas/services/mcp_eval \
  --setenv PATH /usr/local/bin:/usr/bin:/bin \
  --setenv HOME /root \
  --setenv UV_PROJECT_ENVIRONMENT /tmp/mcpatlas_mcp_eval_venv \
  --setenv UV_CACHE_DIR /root/.cache/uv \
  --setenv MCP_SERVER_URL http://127.0.0.1:1984 \
  --setenv SERVER_URL http://127.0.0.1:3000 \
  --setenv HOST 127.0.0.1 \
  --setenv PORT 3000 \
  --setenv LOG_LEVEL INFO \
  /bin/bash -lc 'uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python -m mcp_completion.main'
```

Run completion and scoring inside the same `bwrap`/`dotenv` pattern, with these inner commands:

```bash
uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py \
  --model openai/alias-code \
  --input /tmp/mcpatlas_inputs/baseline_689bd255c0422b257e7dfcc5.csv \
  --output hpc_smoke_baseline_alias_code_20260702.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter

uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py \
  --model openai/alias-code \
  --input /tmp/mcpatlas_inputs/perturbed_689bd255c0422b257e7dfcc5.csv \
  --output hpc_smoke_perturbed_alias_code_20260702.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter

uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py \
  --input-file completion_results/hpc_smoke_baseline_alias_code_20260702.csv \
  --model-label hpc_smoke_baseline_alias_code_20260702 \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1

uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py \
  --input-file completion_results/hpc_smoke_perturbed_alias_code_20260702.csv \
  --model-label hpc_smoke_perturbed_alias_code_20260702 \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1
```

## Known Blockers And Fixes

| blocker | fix |
|---|---|
| Docker unavailable | Use extracted Apptainer image rootfs plus `bwrap` |
| Direct SIF run fails because `/dev/fuse` is unavailable | Continue using extracted rootfs |
| Host Python is too old for `mcp_eval` | Use the official image Python 3.12 and image-local `uv` |
| Secret `.env` has CRLF line endings | Load with `python-dotenv`; do not shell-source it |
| Perturbation score did not drop | Removing `mcp-code-executor_execute_code` produced a score drop to `0.0`; use that perturbation as the one-task capability-gap smoke |

## Minimal Next Team Instruction

For routine team use on this host:

1. Start the agent service with `filesystem,mcp-code-executor`.
2. Start `mcp_eval` through the `bwrap`/`dotenv` command.
3. Use one-row CSVs under `/tmp/mcpatlas_eval_tmp/mcpatlas_inputs/`.
4. Run completion with `openai/alias-code`.
5. Score with `openai/alias-ha`.
6. Report only summary fields unless raw outputs have been reviewed for sharing.
