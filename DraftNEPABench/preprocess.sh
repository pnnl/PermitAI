#!/usr/bin/env bash

# Preprocess tasks by running the utility scripts in order:
# 1) utils/rename_task_folders.py
# 2) utils/prepare_workdirs.py
# 3) utils/docx_to_md.py
# 4) utils/update_references.py
# 5) utils/move_evaluation_documents.py
#
# Usage:
#   ./preprocess.sh [TASKS]
#   ./preprocess.sh --task TASKS   # also accepted
#   ./preprocess.sh --tasks TASKS  # also accepted
#
# TASKS can be:
#   - "all" (default)
#   - a single id (e.g., "3"), name (e.g., "task-3")
#   - a comma-separated list (e.g., "1,3,5" or "task-1,task-3")

set -euo pipefail

# Accept either positional TASKS or --task/--tasks TASKS
TASKS_SPEC="all"
if [[ $# -ge 1 ]]; then
  case "$1" in
    --task|--tasks)
      if [[ $# -ge 2 ]]; then
        TASKS_SPEC="$2"
      else
        echo "Error: $1 expects a value (e.g., 1,3,5 or task-1,task-3)" >&2
        exit 2
      fi
      ;;
    *)
      TASKS_SPEC="$1"
      ;;
  esac
fi

echo "[1/5] Renaming task folders (task_ -> task-)"
python3 utils/rename_task_folders.py

echo "[2/5] Preparing workdirs for tasks: ${TASKS_SPEC}"
if [[ "${TASKS_SPEC}" == "all" ]]; then
  python3 utils/prepare_workdirs.py
else
  # prepare_workdirs.py expects --task (singular)
  python3 utils/prepare_workdirs.py --task "${TASKS_SPEC}"
fi

echo "[3/5] Converting DOCX to Markdown for tasks: ${TASKS_SPEC}"
python3 utils/docx_to_md.py --tasks "${TASKS_SPEC}"

echo "[4/5] Updating References for tasks: ${TASKS_SPEC}"
python3 utils/update_references.py --tasks "${TASKS_SPEC}"

echo "[5/5] Moving evaluation documents"
if [[ "${TASKS_SPEC}" == "all" ]]; then
  python3 utils/move_evaluation_documents.py
else
  # Loop through each task
  IFS="," read -r -a TOKENS <<< "${TASKS_SPEC}"
  for tok in "${TOKENS[@]}"; do
    tok_trimmed="${tok//[[:space:]]/}"
    if [[ -z "${tok_trimmed}" ]]; then
      continue
    fi
    if [[ "${tok_trimmed}" =~ ^[0-9]+$ ]]; then
      canon="task-${tok_trimmed}"
    elif [[ "${tok_trimmed}" =~ ^task_[0-9]+$ ]]; then
      canon="task-${tok_trimmed#task_}"
    else
      canon="${tok_trimmed}"
    fi
    echo "  - Moving for ${canon}"
    python3 utils/move_evaluation_documents.py --only "${canon}"
  done
fi

echo "Done."
