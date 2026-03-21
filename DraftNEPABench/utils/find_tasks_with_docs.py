#!/usr/bin/env python3
"""
Scan `./run-and-grade/tasks` for task directories. For each immediate task
directory, search recursively within all nested subfolders for either
`ground_truth.docx` or `rubric.md`. Print the matching task directory paths
(one per line).

"""

from pathlib import Path
import sys


def resolve_default_tasks_root() -> Path:
    # Repository root is two levels up from this script: `utils/` -> repo root
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "run-and-grade" / "tasks"


def find_matching_tasks(tasks_root: Path) -> list[Path]:
    matches: list[Path] = []
    if not tasks_root.exists():
        return matches

    # Iterate immediate task directories, but search recursively within each
    for task_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        has_docs = any(task_dir.rglob("ground_truth.docx")) or any(
            task_dir.rglob("rubric.md")
        )
        if has_docs:
            matches.append(task_dir)
    return matches


def main() -> int:
    if len(sys.argv) > 1:
        tasks_root = Path(sys.argv[1]).resolve()
    else:
        tasks_root = resolve_default_tasks_root()

    if not tasks_root.exists():
        print(f"Tasks root not found: {tasks_root}", file=sys.stderr)
        return 1

    matches = find_matching_tasks(tasks_root)
    for m in matches:
        print(m.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
