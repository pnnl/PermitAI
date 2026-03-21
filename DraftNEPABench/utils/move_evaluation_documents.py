#!/usr/bin/env python3
"""
Move evaluation documents from each task into a central folder.

For each task directory under `run-and-grade/tasks/task-*`, this script moves:
 - `rubric.md`
 - `workdir/task_documents` directory

Destination layout:
 `run-and-grade/evaluation-documents/<task-id>/rubric.md`
 `run-and-grade/evaluation-documents/<task-id>/task_documents/`

Usage:
  - Run for all tasks:            python utils/move_evaluation_documents.py
  - Run for a single task:        python utils/move_evaluation_documents.py --only task-3
  - Overwrite existing outputs:   python utils/move_evaluation_documents.py --force

Use `--force` to overwrite existing destination by removing them first.
"""

import argparse
import os
import shutil
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TASKS_ROOT = os.path.join(BASE_DIR, "run-and-grade", "tasks")
EVAL_ROOT = os.path.join(BASE_DIR, "run-and-grade", "evaluation-documents")


def move_file(src: str, dst: str, force: bool = False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        if force:
            if os.path.isdir(dst) and not os.path.islink(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        else:
            print(f"[skip] Destination exists: {dst}")
            return
    print(f"[move] {src} -> {dst}")
    shutil.move(src, dst)


def move_dir(src: str, dst: str, force: bool = False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[skip] Destination dir exists: {dst}")
            return
    print(f"[move] {src} -> {dst}")
    shutil.move(src, dst)


def process_task(task_dir: str, force: bool = False) -> None:
    task_id = os.path.basename(task_dir)
    if not os.path.isdir(task_dir):
        return

    # Source paths
    rubric_src = os.path.join(task_dir, "rubric.md")
    task_docs_src = os.path.join(task_dir, "workdir", "task_documents")

    # Destination paths
    task_eval_root = os.path.join(EVAL_ROOT, task_id)
    rubric_dst = os.path.join(task_eval_root, "rubric.md")
    task_docs_dst = os.path.join(task_eval_root, "task_documents")

    moved_any = False

    if os.path.isfile(rubric_src):
        move_file(rubric_src, rubric_dst, force=force)
        moved_any = True
    else:
        print(f"[miss] No rubric.md in {task_dir}")

    if os.path.isdir(task_docs_src):
        move_dir(task_docs_src, task_docs_dst, force=force)
        moved_any = True
    else:
        print(f"[miss] No workdir/task_documents in {task_dir}")

    if not moved_any:
        print(f"[note] Nothing to move for {task_id}")


def discover_tasks(root: str):
    for name in sorted(os.listdir(root)):
        if name.startswith("task-"):
            yield os.path.join(root, name)


def main():
    parser = argparse.ArgumentParser(description="Move evaluation documents from tasks to a central folder")
    parser.add_argument("--only", metavar="TASK_ID", help="Only process a single task (e.g., task-3)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination contents")
    args = parser.parse_args()

    if not os.path.isdir(TASKS_ROOT):
        print(f"[error] Tasks root not found: {TASKS_ROOT}")
        return 1

    os.makedirs(EVAL_ROOT, exist_ok=True)

    if args.only:
        task_dir = os.path.join(TASKS_ROOT, args.only)
        if not os.path.isdir(task_dir):
            print(f"[error] Task not found: {task_dir}")
            return 1
        process_task(task_dir, force=args.force)
    else:
        for task_dir in discover_tasks(TASKS_ROOT):
            process_task(task_dir, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())

