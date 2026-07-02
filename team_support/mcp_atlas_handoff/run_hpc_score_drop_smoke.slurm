#!/bin/bash
#SBATCH --job-name=mcpatlas-smoke
#SBATCH --output=/tmp/mcpatlas_smoke_%j.out
#SBATCH --error=/tmp/mcpatlas_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# This job sends the selected MCP-Atlas task prompt, generated tool outputs, and
# evaluator payloads to the configured LLM endpoint. Run only after approval.

TASK_ID="${TASK_ID:-689bd255c0422b257e7dfcc5}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"

ROOT="${ROOT:-/tmp/mcpatlas_agent_tree}"
REPO="${REPO:-/home/omel305g/masters_workspace/mcp-atlas}"
SECRET_ENV="${SECRET_ENV:-$REPO/services/mcp_eval/.env}"
EVAL_TMP="${EVAL_TMP:-/tmp/mcpatlas_eval_tmp}"
UV_CACHE="${UV_CACHE:-/tmp/mcpatlas_uv_cache}"
INPUT_DIR="${INPUT_DIR:-$EVAL_TMP/mcpatlas_inputs}"
LOG_DIR="${LOG_DIR:-$EVAL_TMP/logs/$RUN_LABEL}"

COMPLETION_MODEL="${COMPLETION_MODEL:-openai/alias-code}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-openai/alias-ha}"

BASELINE_INPUT="$INPUT_DIR/baseline_${TASK_ID}.csv"
NO_CODE_INPUT="$INPUT_DIR/perturbed_no_code_executor_${TASK_ID}.csv"

BASELINE_LABEL="hpc_batch_baseline_alias_code_${RUN_LABEL}"
NO_CODE_LABEL="hpc_batch_no_code_executor_alias_code_${RUN_LABEL}"

AGENT_PID=""
EVAL_PID=""

cleanup() {
  set +e
  if [ -n "$EVAL_PID" ]; then
    kill "$EVAL_PID" >/dev/null 2>&1 || true
    wait "$EVAL_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$AGENT_PID" ]; then
    kill "$AGENT_PID" >/dev/null 2>&1 || true
    wait "$AGENT_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_path() {
  if [ ! -e "$1" ]; then
    echo "missing required path: $1" >&2
    exit 1
  fi
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"
  local delay="${4:-2}"
  local i

  for i in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      echo "$name ready"
      return 0
    fi
    sleep "$delay"
  done

  echo "$name did not become ready at $url" >&2
  return 1
}

eval_bwrap() {
  local inner="$1"
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
    --setenv PYTHONUTF8 1 \
    --setenv PYTHONIOENCODING utf-8 \
    /bin/bash -lc "$inner"
}

prepare_inputs() {
  mkdir -p "$INPUT_DIR"
  python - "$REPO/services/mcp_eval/sample_tasks.csv" "$BASELINE_INPUT" "$NO_CODE_INPUT" "$TASK_ID" <<'PY'
import csv
import json
import pathlib
import sys

sample = pathlib.Path(sys.argv[1])
baseline_out = pathlib.Path(sys.argv[2])
no_code_out = pathlib.Path(sys.argv[3])
task_id = sys.argv[4]

with sample.open(newline="") as f:
    rows = list(csv.DictReader(f))

row = next((r for r in rows if r["TASK"] == task_id), None)
if row is None:
    raise SystemExit(f"task not found in sample_tasks.csv: {task_id}")

fields = list(rows[0].keys())
baseline_out.parent.mkdir(parents=True, exist_ok=True)

with baseline_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(row)

perturbed = dict(row)
tools = json.loads(perturbed["ENABLED_TOOLS"])
perturbed["ENABLED_TOOLS"] = json.dumps(
    [t for t in tools if t != "mcp-code-executor_execute_code"]
)

with no_code_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(perturbed)

print(f"prepared baseline input: {baseline_out}")
print(f"prepared no-code-executor input: {no_code_out}")
print(f"baseline tools: {len(tools)}")
print(f"no-code-executor tools: {len(json.loads(perturbed['ENABLED_TOOLS']))}")
PY
}

write_summary() {
  python - "$REPO/services/mcp_eval" "$BASELINE_LABEL" "$NO_CODE_LABEL" "$LOG_DIR/summary.tsv" <<'PY'
import csv
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
baseline_label = sys.argv[2]
no_code_label = sys.argv[3]
summary_path = pathlib.Path(sys.argv[4])

def mean_score(label):
    path = base / "evaluation_results" / f"coverage_stats_{label}.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stat") == "mean coverage score":
                return row.get("value", "")
    return ""

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w") as f:
    f.write("run\tremoved_tool\tscore\n")
    f.write(f"baseline\tnone\t{mean_score(baseline_label)}\n")
    f.write(
        "no-code-executor\tmcp-code-executor_execute_code\t"
        f"{mean_score(no_code_label)}\n"
    )

print(f"safe summary: {summary_path}")
print(summary_path.read_text())
PY
}

echo "MCP-Atlas score-drop smoke job"
echo "task: $TASK_ID"
echo "run label: $RUN_LABEL"
echo "logs: $LOG_DIR"

command -v bwrap >/dev/null
command -v curl >/dev/null
require_path "$ROOT"
require_path "$REPO/services/mcp_eval/sample_tasks.csv"
require_path "$SECRET_ENV"
mkdir -p "$LOG_DIR" "$UV_CACHE" "$EVAL_TMP"

prepare_inputs

echo "starting agent service"
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
  /agent-environment/.venv/bin/python -m uvicorn agent_environment.main:app \
    --host 127.0.0.1 --port 1984 \
  >"$LOG_DIR/agent.log" 2>&1 &
AGENT_PID="$!"

wait_for_url "agent" "http://127.0.0.1:1984/enabled-servers"

echo "starting mcp_eval service"
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
  /bin/bash -lc \
    'uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python -m mcp_completion.main' \
  >"$LOG_DIR/mcp_eval.log" 2>&1 &
EVAL_PID="$!"

wait_for_url "mcp_eval" "http://127.0.0.1:3000/health"

echo "running baseline completion"
eval_bwrap "uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py --model '$COMPLETION_MODEL' --input '/tmp/mcpatlas_inputs/baseline_${TASK_ID}.csv' --output '${BASELINE_LABEL}.csv' --num-tasks 1 --concurrency 1 --no-filter"

echo "running no-code-executor completion"
eval_bwrap "uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_completion_script.py --model '$COMPLETION_MODEL' --input '/tmp/mcpatlas_inputs/perturbed_no_code_executor_${TASK_ID}.csv' --output '${NO_CODE_LABEL}.csv' --num-tasks 1 --concurrency 1 --no-filter"

echo "scoring baseline"
eval_bwrap "uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py --input-file 'completion_results/${BASELINE_LABEL}.csv' --model-label '$BASELINE_LABEL' --evaluator-model '$EVALUATOR_MODEL' --num-tasks 1 --concurrency 1"

echo "scoring no-code-executor"
eval_bwrap "uv run --offline dotenv -f /run/secrets/mcp_atlas.env run -- python mcp_evals_scores.py --input-file 'completion_results/${NO_CODE_LABEL}.csv' --model-label '$NO_CODE_LABEL' --evaluator-model '$EVALUATOR_MODEL' --num-tasks 1 --concurrency 1"

write_summary

echo "completion outputs:"
echo "$REPO/services/mcp_eval/completion_results/${BASELINE_LABEL}.csv"
echo "$REPO/services/mcp_eval/completion_results/${NO_CODE_LABEL}.csv"
echo "evaluation outputs:"
echo "$REPO/services/mcp_eval/evaluation_results/scored_${BASELINE_LABEL}.csv"
echo "$REPO/services/mcp_eval/evaluation_results/scored_${NO_CODE_LABEL}.csv"
echo "done"
