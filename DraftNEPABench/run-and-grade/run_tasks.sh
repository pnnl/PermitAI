#!/usr/bin/env bash

# Simple runner for task_* directories using Codex CLI by default,
# with optional support for Gemini and Claude coding agents.
# Usage:
#   bash run_tasks.sh --task all [--agent codex|gemini|claude] [--model <name>] [--agent-cmd <template>] [--trials <n>]
#   bash run_tasks.sh --task 1 --agent codex --trials 3
#   bash run_tasks.sh --task 1,2 --agent gemini --model gemini-1.5-pro --trials 5

set -o pipefail

# Resolve directory of this script so paths work from any CWD
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

print_usage() {
  echo "Usage: $0 --task all|<num>[,<num>...] [--agent codex|gemini|claude] [--model <name>] [--agent-cmd <template>] [--trials <n>] [--skip-existing[=true|false]]"
  echo ""
  echo "Options:"
  echo "  --agent       Which agent CLI to use (default: codex)"
  echo "  --model       Model identifier to pass to the chosen agent"
  echo "  --agent-cmd   Command template for non-Codex agents (use {instructions_file})"
  echo "  --trials      Number of times to run each task (default: 1)"
  echo "  --skip-existing   Skip trials only when out/report.md exists (default: true). Use --no-skip-existing or --skip-existing=false to rerun."
}

TASK_ARG="all"
AGENT="codex"
MODEL=""
AGENT_CMD_TEMPLATE=""
TRIALS=1
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires an argument" >&2; print_usage; exit 2; }
      TASK_ARG="$2"; shift 2;
      ;;
    --task=*)
      TASK_ARG="${1#*=}"; shift;
      ;;
    --agent)
      [[ $# -ge 2 ]] || { echo "--agent requires an argument" >&2; print_usage; exit 2; }
      AGENT="$2"; shift 2;
      ;;
    --agent=*)
      AGENT="${1#*=}"; shift;
      ;;
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires an argument" >&2; print_usage; exit 2; }
      MODEL="$2"; shift 2;
      ;;
    --model=*)
      MODEL="${1#*=}"; shift;
      ;;
    --agent-cmd)
      [[ $# -ge 2 ]] || { echo "--agent-cmd requires an argument" >&2; print_usage; exit 2; }
      AGENT_CMD_TEMPLATE="$2"; shift 2;
      ;;
    --agent-cmd=*)
      AGENT_CMD_TEMPLATE="${1#*=}"; shift;
      ;;
    --trials)
      [[ $# -ge 2 ]] || { echo "--trials requires an argument" >&2; print_usage; exit 2; }
      TRIALS="$2"; shift 2;
      ;;
    --trials=*)
      TRIALS="${1#*=}"; shift;
      ;;
    --skip-existing)
      SKIP_EXISTING=1; shift;
      ;;
    --skip-existing=*)
      val="${1#*=}"; shift;
      val_lc=$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')
      case "$val_lc" in
        1|true|yes|y) SKIP_EXISTING=1 ;;
        0|false|no|n) SKIP_EXISTING=0 ;;
        *) echo "Invalid value for --skip-existing: $val" >&2; print_usage; exit 2 ;;
      esac
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0; shift;
      ;;
    -h|--help)
      print_usage; exit 0;
      ;;
    *)
      echo "Unknown argument: $1" >&2; print_usage; exit 2;
      ;;
  esac
done

tasks=()

get_tasks_root() {
  # Prefer a nested tasks/ directory next to this script
  if compgen -G "$SCRIPT_DIR/tasks/task-*" > /dev/null; then
    echo "$SCRIPT_DIR/tasks"
  else
    echo "$SCRIPT_DIR"
  fi
}

discover_all_tasks() {
  # Collect task-* directories and sort numerically by the suffix (bash 3 compatible)
  tasks=()
  local root
  root=$(get_tasks_root)
  while IFS= read -r d; do
    tasks+=("$d")
  done < <(
    for d in "$root"/task-*; do
      [[ -d "$d" ]] || continue
      # Only keep names like */task-<number>
      base_name=${d##*/}
      if [[ "$base_name" =~ ^task-[0-9]+$ ]]; then
        echo "$d"
      fi
    done | sort -t- -k2,2n
  )
}

select_specific_tasks() {
  local list="$1"
  IFS=',' read -r -a req <<< "$list"
  tasks=()
  local root
  root=$(get_tasks_root)
  for raw in "${req[@]}"; do
    # Trim whitespace
    n="${raw//[[:space:]]/}"
    if [[ -z "$n" ]]; then continue; fi
    if [[ ! "$n" =~ ^[0-9]+$ ]]; then
      echo "Invalid task number: '$raw'" >&2; exit 2
    fi
    d="$root/task-$n"
    if [[ ! -d "$d" ]]; then
      echo "Task not found: $d" >&2; exit 2
    fi
    tasks+=("$d")
  done
}

# Lowercase for comparison in a macOS/POSIX-safe way (bash 3 compatible)
TASK_ARG_LC=$(printf '%s' "$TASK_ARG" | tr '[:upper:]' '[:lower:]')
if [[ "$TASK_ARG_LC" == "all" ]]; then
  discover_all_tasks
else
  select_specific_tasks "$TASK_ARG"
fi

if [[ ${#tasks[@]} -eq 0 ]]; then
  echo "No tasks to process."
  exit 0
fi

results_names=()
results_rcs=()

# Normalize agent name (needs to happen before setting per-agent paths)
AGENT_LC=$(printf '%s' "$AGENT" | tr '[:upper:]' '[:lower:]')
case "$AGENT_LC" in
  codex|gemini|claude) ;; 
  *) echo "Unknown --agent '$AGENT' (expected codex|gemini|claude)" >&2; exit 2;;
esac

# Ensure per-agent logs directory exists
LOG_DIR="$SCRIPT_DIR/logs"
AGENT_LOG_DIR="$LOG_DIR/$AGENT_LC"
mkdir -p "$AGENT_LOG_DIR"
# Resolve to absolute path to be safe across subshell cd's
LOG_DIR_ABS=$(cd "$AGENT_LOG_DIR" && pwd)

# Consolidated CSVs
CSV_PATH="$SCRIPT_DIR/logs/codex_runs.csv"
GEMINI_CSV_PATH="$SCRIPT_DIR/logs/gemini_runs.csv"

# Capture Codex CLI version once for logs (only used for Codex)
CODEX_VERSION=$(codex --version 2>/dev/null || echo "unknown")
# Capture Gemini CLI version for logs
GEMINI_VERSION=$(gemini --version 2>/dev/null || echo "unknown")


for dir in "${tasks[@]}"; do
  echo
  echo "=== Running $dir ==="
  # Use shared instructions from the tasks/ folder; run inside each task's workdir when present
  run_base="$dir"
  if [[ -d "$dir/workdir" ]]; then
    run_base="$dir/workdir"
  fi

  root=$(get_tasks_root)
  # instructions_path="$root/agent_instructions.md"
  instructions_path="$root/agent_instructions.md"
  wrapup_path="$root/wrapup.md"

  # Build combined instructions once; reuse for all agents. For Gemini,
  # pass this prompt text directly instead of re-reading from disk.

  if [[ ! -f "$instructions_path" ]]; then
    echo "[skip] $dir: missing agent_instructions.md (looked in $instructions_path)" >&2
    results_names+=("$dir"); results_rcs+=(2)
    continue
  fi

  instructions=$(<"$instructions_path")
  if [[ -z "$instructions" ]]; then
    echo "[skip] $dir: agent_instructions.md is empty (at $instructions_path)" >&2
    results_names+=("$dir"); results_rcs+=(2)
    continue
  fi

  # If wrapup.md exists in the shared tasks/ folder, append it to the instructions
  if [[ -f "$wrapup_path" ]]; then
    wrapup=$(<"$wrapup_path")
    if [[ -n "$wrapup" ]]; then
      instructions+=$'\n\n---\n\n# Wrapup\n\n'
      instructions+="$wrapup"
    fi
  fi

  # Agent-specific extra instructions: e.g., gemini-instructions.md or claude-instructions.md
  agent_extra_path="$root/${AGENT_LC}_instructions.md"
  if [[ -f "$agent_extra_path" ]]; then
    extra=$(<"$agent_extra_path")
    if [[ -n "$extra" ]]; then
      instructions+=$'\n\n---\n\n# Agent Notes ('"$AGENT_LC"')\n\n'
      instructions+="$extra"
    fi
  fi

  echo "cwd: $run_base"

  # Task identifiers, log paths, and per-agent results dir
  task_name=$(basename "$dir")
  TASK_RESULTS_DIR="$SCRIPT_DIR/results/$AGENT_LC/$task_name"
  mkdir -p "$TASK_RESULTS_DIR"

 

  # Run trials for this task
  # Validate TRIALS is a positive integer
  if ! [[ "$TRIALS" =~ ^[0-9]+$ ]] || [[ "$TRIALS" -lt 1 ]]; then
    echo "Invalid --trials '$TRIALS' (must be a positive integer)" >&2
    results_names+=("$dir [$AGENT_LC] trial-1")
    results_rcs+=(2)
    continue
  fi

  for ((trial=1; trial<=TRIALS; trial++)); do
    trial_tag="trial-${trial}"
    TRIAL_DIR="$TASK_RESULTS_DIR/$trial_tag"
    # Agent-specific log file path (include trial tag)
    case "$AGENT_LC" in
      codex) log_path="$LOG_DIR_ABS/${task_name}-codex-${trial_tag}.jsonl" ;;
      gemini) log_path="$LOG_DIR_ABS/${task_name}-gemini-${trial_tag}.out" ;;
      claude) log_path="$LOG_DIR_ABS/${task_name}-claude-${trial_tag}.out" ;;
    esac
    last_msg_path="$LOG_DIR_ABS/${task_name}-last-${trial_tag}.md"
    stats_output_path="${log_path%.*}-stats.json"

    # skip only if there is an out folder with report.md
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
      if [[ -f "$TRIAL_DIR/out/report.md" ]] || [[ -f "$run_base/out/report.md" ]]; then
        echo "[skip] $dir [$AGENT_LC] $trial_tag: out/report.md exists"
        results_names+=("$dir [$AGENT_LC] $trial_tag"); results_rcs+=(0)
        continue
      fi
    fi

    # Record start time (ISO 8601 UTC and epoch seconds)
    start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    start_epoch=$(date +%s)

    # Run the selected agent and capture output
    (
      cd "$run_base" || exit 3
      case "$AGENT_LC" in
        codex)
          if ! command -v codex >/dev/null 2>&1; then
            echo "codex CLI not found in PATH" >&2; exit 127
          fi
          codex exec --full-auto -m gpt-5 -s danger-full-access --json \
            --output-last-message "$last_msg_path" \
            "$instructions" --config model_reasoning_effort=high \
            >"$log_path" 2>"$LOG_DIR_ABS/${task_name}-codex-${trial_tag}.stderr"
          ;;
        gemini)
          if ! command -v gemini >/dev/null 2>&1; then
            echo "gemini CLI not found in PATH" >&2; exit 127
          fi

          TELEMETRY_FILE="${LOG_DIR_ABS}/${task_name}-gemini-${trial_tag}.telemetry.log"
          { time gemini agent -m "${MODEL:-gemini-2.5-pro}" --yolo --telemetry --telemetry-target local \
          --telemetry-otlp-endpoint="" --telemetry-outfile="$TELEMETRY_FILE" \
          --debug "$instructions"; } >"$log_path" 2> >(tee "${LOG_DIR_ABS}/${task_name}-gemini-${trial_tag}.stderr")

          ;;
        claude)
        {
          set -x
         claude -p "$instructions" --output-format json --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
         --dangerously-skip-permissions >"$log_path" 2>"$LOG_DIR_ABS/${task_name}-claude-${trial_tag}.stderr"
          
          set +x
        } 2>>"$LOG_DIR_ABS/${task_name}-claude-${trial_tag}.debug.log"

        ;;


      esac
    )
    rc=$?


    # Record end time and compute duration
    end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    end_epoch=$(date +%s)
    duration_sec=$(( end_epoch - start_epoch ))

   
    # Ensure trial directory exists before moving artifacts
    mkdir -p "$TRIAL_DIR"

    # Move outputs and artifacts into per-trial directory
    if [[ -d "$run_base/out" ]]; then
      mkdir -p "$TRIAL_DIR"
      rm -rf "$TRIAL_DIR/out" 2>/dev/null || true
      if mv "$run_base/out" "$TRIAL_DIR/out" 2>/dev/null; then
        :
      else
        mkdir -p "$TRIAL_DIR/out"
        cp -R "$run_base/out/." "$TRIAL_DIR/out/" 2>/dev/null || true
        rm -rf "$run_base/out" 2>/dev/null || true
      fi
    fi

    # PLAN.md
    if [[ -f "$run_base/PLAN.md" ]]; then
      mv "$run_base/PLAN.md" "$TRIAL_DIR/PLAN.md" 2>/dev/null || cp "$run_base/PLAN.md" "$TRIAL_DIR/PLAN.md" 2>/dev/null || true
    fi
    # scratchpad
    if [[ -f "$run_base/scratchpad.md" ]]; then
      mv "$run_base/scratchpad.md" "$TRIAL_DIR/scratchpad.md" 2>/dev/null || cp "$run_base/scratchpad.md" "$TRIAL_DIR/scratchpad.md" 2>/dev/null || true
    fi
    # text folder
    if [[ -d "$run_base/text" ]]; then
      rm -rf "$TRIAL_DIR/text" 2>/dev/null || true
      if mv "$run_base/text" "$TRIAL_DIR/text" 2>/dev/null; then
        :
      else
        mkdir -p "$TRIAL_DIR/text"
        cp -R "$run_base/text/." "$TRIAL_DIR/text/" 2>/dev/null || true
        rm -rf "$run_base/text" 2>/dev/null || true
      fi
    fi

    # Move all remaining items except References directory and task.md
    if [[ -d "$run_base" ]]; then
      while IFS= read -r -d '' item; do
        base_name=$(basename "$item")
        case "$base_name" in
          References|task.md|out|text|PLAN.md|scratchpad.md)
            continue
            ;;
        esac
        # Prefer move; fall back to copy+remove if cross-device or permission issues
        if mv "$item" "$TRIAL_DIR/" 2>/dev/null; then
          :
        else
          cp -R "$item" "$TRIAL_DIR/" 2>/dev/null || true
          rm -rf "$item" 2>/dev/null || true
        fi
      done < <(find "$run_base" -mindepth 1 -maxdepth 1 -print0)
    fi


    results_names+=("$dir [$AGENT_LC] $trial_tag")
    results_rcs+=("$rc")
  done
done

echo
echo "=== Summary ==="
for i in "${!results_names[@]}"; do
  name="${results_names[$i]}"
  rc="${results_rcs[$i]}"
  if [[ "$rc" -eq 0 ]]; then
    echo "- $name: ok"
  else
    echo "- $name: fail (rc=$rc)"
  fi
done

echo
echo "All requested tasks processed."

#--sandbox workspace-write --ask-for-approval never