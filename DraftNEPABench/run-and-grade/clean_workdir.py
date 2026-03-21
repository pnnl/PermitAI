from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List


ALLOWED_FILES = {"task.md", "wrapup.md"}
REFERENCES_DIR = "References"


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean task workdir, keeping only references PDFs and key markdown files.")
    p.add_argument("--task", required=True, help="'all' or number(s) like '1' or '1,2'")
    p.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    p.add_argument("--verbose", action="store_true", help="Print kept and scanned paths")
    return p.parse_args(list(argv))


def get_tasks_root(base: Path) -> Path:
    tasks_dir = base / "tasks"
    try:
        has_nested = any(p.is_dir() for p in tasks_dir.glob("task-*"))
    except Exception:
        has_nested = False
    return tasks_dir if tasks_dir.is_dir() and has_nested else base


def discover_tasks(base: Path) -> List[Path]:
    tasks: List[Path] = []
    root = get_tasks_root(base)
    for p in sorted(root.glob("task-*"), key=lambda x: int(re.sub(r"^task-", "", x.name)) if re.match(r"^task-\d+$", x.name) else float("inf")):
        if p.is_dir() and re.match(r"^task-\d+$", p.name):
            tasks.append(p)
    return tasks


def select_tasks(base: Path, spec: str) -> List[Path]:
    spec_lc = spec.strip().lower()
    if spec_lc == "all":
        return discover_tasks(base)
    out: List[Path] = []
    root = get_tasks_root(base)
    for part in [s.strip() for s in spec.split(",") if s.strip()]:
        if not part.isdigit():
            raise SystemExit(f"Invalid task specifier: {part}")
        p = root / f"task-{int(part)}"
        if not p.is_dir():
            raise SystemExit(f"Task not found: {p}")
        out.append(p)
    return out


def remove_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] remove: {path}")
        return
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] failed to remove {path}: {e}", file=sys.stderr)


def clean_references(ref_dir: Path, *, dry_run: bool, verbose: bool) -> None:
    if verbose:
        print(f"Scanning references: {ref_dir}")
    # Remove non-PDF files under references, keep .pdf (case-insensitive)
    for root, dirs, files in os.walk(ref_dir):
        root_path = Path(root)
        for fname in files:
            p = root_path / fname
            if p.suffix.lower() != ".pdf":
                remove_path(p, dry_run=dry_run)
            elif verbose:
                print(f"keep: {p}")
    # Optionally prune empty directories (except the root references dir)
    # Walk bottom-up to remove empties
    for root, dirs, files in os.walk(ref_dir, topdown=False):
        root_path = Path(root)
        if root_path == ref_dir:
            continue
        # If directory becomes empty after file removals, delete it
        try:
            if not any(root_path.iterdir()):
                remove_path(root_path, dry_run=dry_run)
        except Exception:
            pass


def clean_workdir(workdir: Path, *, dry_run: bool, verbose: bool) -> None:
    if not workdir.exists():
        print(f"[skip] no workdir: {workdir}")
        return
    if verbose:
        print(f"Cleaning workdir: {workdir}")

    for entry in workdir.iterdir():
        name = entry.name
        if entry.is_dir():
            if name == REFERENCES_DIR:
                clean_references(entry, dry_run=dry_run, verbose=verbose)
            else:
                remove_path(entry, dry_run=dry_run)
        else:
            if name.lower() in ALLOWED_FILES:
                if verbose:
                    print(f"keep: {entry}")
            else:
                remove_path(entry, dry_run=dry_run)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    # Anchor to the script's directory so it works from any CWD
    base = Path(__file__).resolve().parent
    try:
        task_dirs = select_tasks(base, args.task)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    if not task_dirs:
        print("No tasks to process.")
        return 0

    for tdir in task_dirs:
        wdir = tdir / "workdir"
        print(f"\n=== Cleaning {tdir.name} ===")
        if not wdir.is_dir():
            print(f"[skip] missing workdir: {wdir}")
            continue
        clean_workdir(wdir, dry_run=args.dry_run, verbose=args.verbose)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
