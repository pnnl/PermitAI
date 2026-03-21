"""
Prepare task directories by ensuring a workdir layout and moving assets.

For each task directory (e.g., task-3):
  - Ensure a "workdir" directory exists
  - If a top-level "References" (or "references") directory exists, move/merge it into workdir/
  - If ALL three files exist at the task root: ground_truth.docx, rubric.docx, task.docx
      -> create workdir/task_documents and move these files into it

Usage:
  python utils/prepare_workdirs.py [--dry-run] [--tasks TASKS] [--tasks-root PATH]

Examples:
  # All tasks under run-and-grade/tasks
  python utils/prepare_workdirs.py

  # Specific tasks by id or name
  python utils/prepare_workdirs.py --tasks 1,3,5
  python utils/prepare_workdirs.py --tasks task-3,task-7

Notes:
  - Non-destructive: when moving into an existing destination, existing files are kept; conflicting
    filenames are skipped with a warning.
  - "References" is moved into workdir preserving its original case .
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


DOCX_FILENAMES = ("ground_truth.docx", "rubric.docx", "task.docx")


def _default_tasks_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "run-and-grade" / "tasks"


def _resolve_task_dir(tasks_root: Path, token: str) -> Path:
    """Resolve a task token to a directory path.
    Accepts forms: "task-3", "task_3", or numeric "3".
    Prefers dash variant when both exist.
    """
    token = token.strip()
    if token.isdigit():
        dash = tasks_root / f"task-{token}"
        under = tasks_root / f"task_{token}"
        if dash.is_dir():
            return dash
        if under.is_dir():
            return under
    else:
        path = tasks_root / token
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Task not found for token: {token}")


def _iter_all_tasks(tasks_root: Path) -> Iterable[Path]:
    # Prefer dash pattern first for stable ordering, then underscore
    dash = sorted(p for p in tasks_root.glob("task-*") if p.is_dir())
    under = sorted(p for p in tasks_root.glob("task_*") if p.is_dir())
    # Avoid duplicates: if both task-3 and task_3 exist, keep only task-3
    existing_dash_nums = {p.name.split("-", 1)[1] for p in dash if "-" in p.name}
    for p in dash:
        yield p
    for p in under:
        num = p.name.split("_", 1)[1] if "_" in p.name else None
        if num and num in existing_dash_nums:
            continue
        yield p


def _ensure_dir(path: Path, dry: bool):
    if dry:
        if not path.exists():
            print(f"Would create directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _move_dir_contents(src: Path, dst: Path, dry: bool):
    """Move a directory into destination.
    - If dst doesn't exist, perform a rename of the directory for efficiency.
    - If dst exists, move files/dirs inside src into dst; skip conflicts.
    - Remove src if it becomes empty after merging.
    """
    if dry:
        if not src.exists():
            return
        if not dst.exists():
            print(f"Would move directory: {src} -> {dst}")
        else:
            for child in sorted(src.iterdir()):
                target = dst / child.name
                if target.exists():
                    print(f"Would skip existing: {target}")
                else:
                    print(f"Would move: {child} -> {target}")
        return

    if not src.exists():
        return
    if not dst.exists():
        # Fast path: rename directory
        src.rename(dst)
        return
    # Merge contents
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        target = dst / child.name
        if target.exists():
            print(f"Skip (exists): {target}")
            continue
        shutil.move(str(child), str(target))
    # Remove src if empty
    try:
        next(src.iterdir())
    except StopIteration:
        src.rmdir()


def _move_files_into(src_dir: Path, filenames: Iterable[str], dst_dir: Path, dry: bool) -> Tuple[int, int]:
    moved = 0
    skipped = 0
    for name in filenames:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            skipped += 1
            continue
        if dry:
            if dst.exists():
                print(f"Would skip existing: {dst}")
            else:
                print(f"Would move: {src} -> {dst}")
            moved += 1
        else:
            if dst.exists():
                print(f"Skip (exists): {dst}")
                skipped += 1
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
    return moved, skipped


def process_task(task_dir: Path, dry: bool = False) -> int:
    print(f"Processing {task_dir}")
    workdir = task_dir / "workdir"
    _ensure_dir(workdir, dry)

    # Move References or references into workdir
    for name in ("References", "references"):
        src = task_dir / name
        if src.exists() and src.is_dir():
            dst = workdir / name
            _move_dir_contents(src, dst, dry)

    # Move .docx into workdir/task_documents if ALL present at task root
    have_all = all((task_dir / f).is_file() for f in DOCX_FILENAMES)
    if have_all:
        dst_dir = workdir / "task_documents"
        if dry and not dst_dir.exists():
            print(f"Would create directory: {dst_dir}")
        elif not dry:
            dst_dir.mkdir(parents=True, exist_ok=True)
        moved, _ = _move_files_into(task_dir, DOCX_FILENAMES, dst_dir, dry)
        if moved:
            print(f"Moved {moved} docx file(s) into {dst_dir}")
    else:
        missing = [f for f in DOCX_FILENAMES if not (task_dir / f).exists()]
        if missing:
            print(f"Docx not moved (missing at task root): {', '.join(missing)}")
    return 0


def parse_tasks_arg(tasks_root: Path, tasks_arg: str | None) -> List[Path]:
    if not tasks_arg or tasks_arg.strip().lower() == "all":
        return list(_iter_all_tasks(tasks_root))
    # Accept comma-separated tokens
    tokens = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    paths: List[Path] = []
    for t in tokens:
        try:
            paths.append(_resolve_task_dir(tasks_root, t))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description="Ensure workdir structure and move References/docx into place")
    p.add_argument("--dry-run", action="store_true", help="Print actions without making changes")
    p.add_argument("--task", default=None, help="'all' or comma-separated list like '1,3,5' or 'task-1,task-3'")
    p.add_argument("--tasks-root", type=Path, default=None, help="Path to tasks root (defaults to repo run-and-grade/tasks)")
    args = p.parse_args(argv)

    tasks_root = args.tasks_root or _default_tasks_root()
    if not tasks_root.is_dir():
        print(f"Error: tasks root not found: {tasks_root}", file=sys.stderr)
        return 2

    task_dirs = parse_tasks_arg(tasks_root, args.task)
    if not task_dirs:
        print("No tasks matched; nothing to do.")
        return 0

    rc = 0
    for td in task_dirs:
        try:
            process_task(td, dry=args.dry_run)
        except Exception as e:
            print(f"Error processing {td}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

