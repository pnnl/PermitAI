"""
Rename task folders from "task_<N>" to "task-<N>" under run-and-grade/tasks.

Usage:
  python utils/rename_task_folders.py [--dry-run] [--tasks-root PATH]

Defaults:
  --tasks-root auto-detects "../run-and-grade/tasks" relative to this script.
  When --dry-run is set, only prints the planned renames.

Behavior:
  - Only renames directories matching ^task_(\d+)$
  - Skips if destination already exists, printing a warning
  - Prints a summary at the end
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TASK_UNDERSCORE_RE = re.compile(r"^task_(\d+)$")


def _default_tasks_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "run-and-grade" / "tasks"


def find_candidates(tasks_root: Path):
    for child in tasks_root.iterdir():
        if not child.is_dir():
            continue
        m = TASK_UNDERSCORE_RE.match(child.name)
        if not m:
            continue
        n = m.group(1)
        dest = child.parent / f"task-{n}"
        yield child, dest


def rename_all(tasks_root: Path, dry_run: bool = False) -> int:
    if not tasks_root.is_dir():
        print(f"Error: tasks root not found: {tasks_root}", file=sys.stderr)
        return 2

    planned = list(find_candidates(tasks_root))
    if not planned:
        print("No task_<N> folders found; nothing to do.")
        return 0

    renamed = 0
    skipped = 0
    for src, dst in sorted(planned, key=lambda t: t[0].name):
        if dst.exists():
            print(f"Skip: {src.name} -> {dst.name} (destination exists)")
            skipped += 1
            continue
        if dry_run:
            print(f"Would rename: {src.name} -> {dst.name}")
        else:
            src.rename(dst)
            print(f"Renamed: {src.name} -> {dst.name}")
        renamed += 1

    print(f"Summary: {renamed} planned/renamed, {skipped} skipped.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Rename task_<N> to task-<N> under run-and-grade/tasks")
    p.add_argument("--dry-run", action="store_true", help="Only print planned renames")
    p.add_argument("--tasks-root", type=Path, default=None, help="Path to tasks root (defaults to repo run-and-grade/tasks)")
    args = p.parse_args(argv)

    tasks_root = args.tasks_root or _default_tasks_root()
    return rename_all(tasks_root, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

