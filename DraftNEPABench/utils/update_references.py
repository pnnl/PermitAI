"""
Update References links in Markdown to point to local files.

Two modes are supported:
1) Task mode (preferred): give a task name like "task-3". The script:
   - Locates `run-and-grade/tasks/task-3/workdir/task.md`.
   - Looks for PDFs in `run-and-grade/tasks/task-3/workdir/references/`.
   - In the References section, replaces external links with plain paths to
     the matching local PDF. Leaves links untouched if no matching PDF exists.
   - If there are extra PDFs in the references folder that are not mentioned in
     the References section of the Markdown, append them as new entries at the
     end of the References section.

2) Case mode (legacy): give a case name like "case1".
   - Locates Markdown in `cases/<case>/` (expects exactly one .md).
   - Looks for files in `references/<case>/`.
   - Replaces links to local relative paths.

Matching strategy:
- First match by URL basename against files in the references folder (case-insensitive).
- If not found, attempt to match using the link label and description text:
  - Extract explicit '*.pdf' filenames if they appear in the text.
  - Otherwise, use token overlap between description and local filenames.
- If still no match, the original link is left unchanged.

No external dependencies.
"""
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from urllib.parse import urlparse, unquote


def find_case_markdown(case: str) -> Path:
    case_dir = Path('cases') / case
    if not case_dir.exists() or not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    md_files = sorted(p for p in case_dir.iterdir() if p.suffix.lower() == '.md')
    if len(md_files) == 0:
        raise FileNotFoundError(f"No .md file found in {case_dir}")
    if len(md_files) > 1:
        names = ', '.join(p.name for p in md_files)
        raise RuntimeError(f"Multiple .md files found in {case_dir}: {names}. Please keep only one.")
    return md_files[0]


def collect_reference_files(case: str) -> list[Path]:
    ref_dir = Path('references') / case
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"References directory not found: {ref_dir}")
    return sorted([p for p in ref_dir.iterdir() if p.is_file()])


# ----- Task mode helpers -----
def find_task_markdown(task_name: str) -> Path:
    task_dir = Path('run-and-grade') / 'tasks' / task_name / 'workdir'
    md_path = task_dir / 'task.md'
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found for task '{task_name}': {md_path}")
    return md_path


def collect_task_reference_files(task_name: str) -> list[Path]:
    base = Path('run-and-grade') / 'tasks' / task_name / 'workdir'
    candidates = [base / 'references', base / 'References']
    ref_paths: list[Path] = []
    for ref_dir in candidates:
        if ref_dir.exists() and ref_dir.is_dir():
            ref_paths.extend([p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() == '.pdf'])
    if not ref_paths:
        raise FileNotFoundError(f"References directory not found for task '{task_name}': {candidates[0]} or {candidates[1]}")
    return sorted(ref_paths)


LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Detect a bare path at end of line, commonly appended in plain mode (case-insensitive)
PLAIN_PATH_RE = re.compile(r"(?P<path>(?:\.{1,2}/)*(?:[Rr]eferences)/\S+)$", re.IGNORECASE)
# Detect any references path occurrence anywhere in a line (case-insensitive)
ANY_REF_PATH_RE = re.compile(r"((?:\.{1,2}/)*(?:[Rr]eferences)/\S+)", re.IGNORECASE)
# Detect a References header in various common formats
REF_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*references\s*$|\*\*references\*\*:??\s*$|references:??\s*$)",
    re.IGNORECASE,
)
# Detect generic section headers to mark the end of the References block
ANY_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*\S.*$|\*\*[^*]+\*\*:??\s*$)",
    re.IGNORECASE,
)


def extract_url_basename(url: str) -> str:
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path or '')
        name = Path(path).name
        return name
    except Exception:
        return ''


def _match_ref_file(base: str, ref_files: list[Path]) -> Path | None:
    if not base:
        return None
    # Exact case-insensitive filename match
    for f in ref_files:
        if f.name.lower() == base.lower():
            return f
    # If base has no extension but one file matches prefix, prefer .pdf
    base_no_q = base.split('?')[0]
    base_stem = Path(base_no_q).stem
    candidates = [f for f in ref_files if f.stem.lower() == base_stem.lower()]
    if candidates:
        return candidates[0]
    # Fallback: substring match
    for f in ref_files:
        if base.lower() in f.name.lower():
            return f
    return None


_FILENAME_IN_TEXT_RE = re.compile(r"([\w .,'\-()+/&]+?\.pdf)\b", re.IGNORECASE)


def _extract_pdf_name_from_text(text: str) -> str | None:
    m = _FILENAME_IN_TEXT_RE.search(text or '')
    if m:
        return ' '.join(m.group(1).strip().split())
    return None


def _tokenize(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or '').lower()) if len(t) >= 3]


def _best_fuzzy_match(text: str, ref_files: list[Path], min_score: float = 0.45) -> Path | None:
    desc_tokens = set(_tokenize(text))
    if not desc_tokens:
        return None
    best = None
    best_score = 0.0
    for f in ref_files:
        fname_tokens = set(_tokenize(f.stem))
        if not fname_tokens:
            continue
        inter = len(desc_tokens & fname_tokens)
        if inter == 0:
            continue
        score = inter / max(1, len(fname_tokens))
        if score > best_score:
            best_score = score
            best = f
    if best and best_score >= min_score:
        return best
    return None


def update_links_globally(
    md_path: Path,
    ref_files: list[Path],
    *,
    as_plain: bool = True,
    from_root: bool = False,
) -> tuple[str, list[str]]:
    """Replace markdown links across the entire document when a matching local
    PDF exists in ref_files. Leave unmatched links as-is. Uses basename, then
    filename-in-text, then token overlap as fallbacks.
    """
    original_text = md_path.read_text(encoding='utf-8')
    notes: list[str] = []

    def compute_rel(target: Path) -> str:
        if from_root:
            # Find the references directory component case-insensitively
            lower_parts = [p.lower() for p in target.parts]
            try:
                i = lower_parts.index('references')
                return '/'.join(target.parts[i:])
            except ValueError:
                return str(target).replace('\\', '/')
        else:
            import os
            return os.path.relpath(str(target), start=str(md_path.parent))

    out_lines: list[str] = []
    for line in original_text.splitlines():
        new_line = ''
        last = 0
        for m in LINK_RE.finditer(line):
            new_line += line[last:m.start()]
            label, url = m.group(1), m.group(2)
            base = extract_url_basename(url)
            match = _match_ref_file(base, ref_files)
            if match is None:
                pdf_name = _extract_pdf_name_from_text(line) or _extract_pdf_name_from_text(label)
                if pdf_name:
                    match = _match_ref_file(pdf_name, ref_files)
            if match is None:
                match = _best_fuzzy_match(f"{label} {line}", ref_files)
            if match is None:
                new_line += m.group(0)
            else:
                rel = compute_rel(match)
                new_line += rel if as_plain else f'[{label}]({rel})'
            last = m.end()
        new_line += line[last:]
        out_lines.append(new_line)

    new_text = '\n'.join(out_lines) + ('\n' if original_text.endswith('\n') else '')
    # Notes for external links that remain
    for m in LINK_RE.finditer(new_text):
        url = m.group(2)
        if url.startswith('http://') or url.startswith('https://'):
            notes.append(f'Unmatched external link kept: {url}')
    return new_text, notes


def update_references_in_md(
    md_path: Path,
    ref_files: list[Path],
    *,
    as_plain: bool = False,
    from_root: bool = False,
    strict: bool = False,
) -> tuple[str, list[str]]:
    lines = md_path.read_text(encoding='utf-8').splitlines()

    # Locate the References section header line index (robust to formats)
    ref_header_idx = None
    for i, line in enumerate(lines):
        if REF_HEADER_RE.match(line):
            ref_header_idx = i
            break
    if ref_header_idx is None:
        raise RuntimeError('Could not find "**References:**" section header in Markdown.')

    # Determine the span of reference list items (consecutive list lines after header)
    # Skip any blank lines after the header
    start = ref_header_idx + 1
    while start < len(lines) and not lines[start].strip():
        start += 1

    end = start
    while end < len(lines):
        s = lines[end].strip()
        if not s:
            break
        # Stop at next section header (e.g., '# Header' or '**Header:**')
        if ANY_SECTION_HEADER_RE.match(s):
            break
        # Include any non-empty line as part of the references block
        end += 1

    # Prepare matching structures
    remaining_files = list(ref_files)
    assigned_paths: dict[int, Path] = {}
    notes: list[str] = []

    # First pass: match by URL basename (from markdown links or plain paths),
    # then fall back to label/description-based matching.
    for idx in range(start, end):
        line = lines[idx]
        m = LINK_RE.search(line)
        base = ''
        if m:
            url = m.group(2).strip()
            base = extract_url_basename(url)
        else:
            m2 = PLAIN_PATH_RE.search(line)
            if m2:
                candidate = m2.group('path').strip()
                base = Path(candidate).name
        match_i = None
        if base:
            # Find case-insensitive exact match within remaining files
            for i, f in enumerate(remaining_files):
                if f.name.lower() == base.lower():
                    match_i = i
                    break
            if match_i is None:
                # Try contains match on filename
                for i, f in enumerate(remaining_files):
                    if base.lower() in f.name.lower():
                        match_i = i
                        break
        # Fallbacks: filename present in text, then token-based fuzzy
        if match_i is None:
            label = m.group(1).strip() if m else ''
            pdf_from_text = _extract_pdf_name_from_text(line) or _extract_pdf_name_from_text(label)
            if pdf_from_text:
                for i, f in enumerate(remaining_files):
                    if f.name.lower() == pdf_from_text.lower():
                        match_i = i
                        break
                if match_i is None:
                    for i, f in enumerate(remaining_files):
                        if pdf_from_text.lower() in f.name.lower():
                            match_i = i
                            break
        if match_i is None and (m is not None):
            candidate = _best_fuzzy_match(f"{m.group(1)} {line}", remaining_files)
            if candidate is not None:
                for i, f in enumerate(remaining_files):
                    if f == candidate:
                        match_i = i
                        break
        if match_i is not None:
            assigned_paths[idx] = remaining_files.pop(match_i)

    # Optional second pass: assign leftover by order (disabled in strict mode)
    if not strict:
        unmatched_line_idxs = [i for i in range(start, end) if i not in assigned_paths]
        if remaining_files and len(unmatched_line_idxs) == len(remaining_files):
            for i, f in zip(unmatched_line_idxs, remaining_files):
                assigned_paths[i] = f
            remaining_files = []

    # Apply replacements
    def _merge_with_single_space(prefix: str, insertion: str, suffix: str = '') -> str:
        pre = prefix.rstrip()
        if pre and not pre.endswith(' '):
            pre += ' '
        return f"{pre}{insertion}{suffix}"
    for idx in range(start, end):
        line = lines[idx]
        m = LINK_RE.search(line)
        label, url = (m.group(1), m.group(2)) if m else (None, None)
        if idx in assigned_paths:
            # Compute relative path from md to reference file
            target = assigned_paths[idx]
            if from_root:
                # Produce a path starting with 'references/...'
                parts = target.parts
                try:
                    i = parts.index('references')
                    rel = '/'.join(parts[i:])
                except ValueError:
                    rel = str(target).replace('\\', '/')
            else:
                rel = Path.relpath(target, start=md_path.parent) if hasattr(Path, 'relpath') else _relpath(target, md_path.parent)
            # Replace: prefer replacing existing link if present; otherwise replace trailing plain path
            if m:
                if as_plain:
                    # ensure exactly one space before the path
                    lines[idx] = _merge_with_single_space(line[:m.start()], rel, line[m.end():])
                else:
                    replacement = f'[{label}]({rel})'
                    lines[idx] = line[:m.start()] + replacement + line[m.end():]
            else:
                # Replace plain path occurrence at end
                m2 = PLAIN_PATH_RE.search(line)
                if m2:
                    lines[idx] = _merge_with_single_space(line[:m2.start('path')], rel, line[m2.end('path'):])
                else:
                    # Fallback: append rel with exactly one space
                    lines[idx] = f"{line.rstrip()} {rel}"
        else:
            if url:
                notes.append(f'No local file match for line {idx+1}: {url}')
            else:
                notes.append(f'No local file match for line {idx+1}: (no link found)')

    # After updating existing lines, append any PDFs in references that are not
    # mentioned anywhere in the document under the References section as new list items.
    def _compute_rel_for(target: Path) -> str:
        if from_root:
            lower_parts = [p.lower() for p in target.parts]
            try:
                i = lower_parts.index('references')
                return '/'.join(target.parts[i:])
            except ValueError:
                return str(target).replace('\\', '/')
        else:
            import os
            return os.path.relpath(str(target), start=str(md_path.parent))

    # Build sets of mentioned reference basenames
    # a) within the References section
    mentioned_in_section: set[str] = set()
    for idx in range(start, end):
        ln = lines[idx]
        for m in LINK_RE.finditer(ln):
            mentioned_in_section.add(extract_url_basename(m.group(2)).lower())
        m2 = PLAIN_PATH_RE.search(ln)
        if m2:
            mentioned_in_section.add(Path(m2.group('path')).name.lower())
        for m3 in ANY_REF_PATH_RE.finditer(ln):
            try:
                mentioned_in_section.add(Path(m3.group(1)).name.lower())
            except Exception:
                pass
    # b) anywhere before the References section
    mentioned_before_section: set[str] = set()
    for idx in range(0, ref_header_idx):
        ln = lines[idx]
        for m in LINK_RE.finditer(ln):
            mentioned_before_section.add(extract_url_basename(m.group(2)).lower())
        m2 = PLAIN_PATH_RE.search(ln)
        if m2:
            mentioned_before_section.add(Path(m2.group('path')).name.lower())
        for m3 in ANY_REF_PATH_RE.finditer(ln):
            try:
                mentioned_before_section.add(Path(m3.group(1)).name.lower())
            except Exception:
                pass

    # Determine which ref files are not mentioned (includes both unmatched and unmentioned files)
    # First, add files that were never matched to any reference entry
    extras: list[Path] = list(remaining_files)
    # Also add files that are not mentioned anywhere in the document
    for f in ref_files:
        if (f.name.lower() not in mentioned_in_section
            and f.name.lower() not in mentioned_before_section
            and f not in extras):
            extras.append(f)
    if extras:
        # Insert new reference paths at the end of the references block
        insert_at = end
        for f in extras:
            rel = _compute_rel_for(f)
            line = f"- {rel}" if as_plain else f"- [{f.name}]({rel})"
            lines.insert(insert_at, line)
            insert_at += 1
        # Add a blank line after appended entries if not already present
        if insert_at >= len(lines) or lines[insert_at].strip() != '':
            lines.insert(insert_at, '')
    else:
        # If no extras appended but the references block ends with a reference path
        # and the following line is not blank, insert a blank line for separation.
        if end > start:
            # Prefer placing a blank line right after the last path-like reference line
            last_path_idx = None
            for i in range(end - 1, start - 1, -1):
                if ANY_REF_PATH_RE.search(lines[i]) or PLAIN_PATH_RE.search(lines[i]):
                    last_path_idx = i
                    break
            if last_path_idx is not None:
                insert_pos = last_path_idx + 1
                if insert_pos >= len(lines) or lines[insert_pos].strip() != '':
                    lines.insert(insert_pos, '')
            else:
                # Fallback: if block ends without a blank and next is non-empty, insert at block end
                if end >= len(lines) or lines[end].strip() != '':
                    lines.insert(end, '')

    new_content = '\n'.join(lines) + ('\n' if lines and not lines[-1].endswith('\n') else '')
    return new_content, notes


def _relpath(target: Path, start: Path) -> str:
    import os
    return os.path.relpath(str(target), start=str(start))


def _tasks_root() -> Path:
    return Path('run-and-grade') / 'tasks'


def _iter_all_tasks(tasks_root: Path):
    dash = sorted([p for p in tasks_root.glob('task-*') if p.is_dir()])
    under = sorted([p for p in tasks_root.glob('task_*') if p.is_dir()])
    dash_nums = {p.name.split('-', 1)[1] for p in dash if '-' in p.name}
    for p in dash:
        yield p
    for p in under:
        num = p.name.split('_', 1)[1] if '_' in p.name else None
        if num and num in dash_nums:
            continue
        yield p


def _resolve_task_dir_any(tasks_root: Path, token: str):
    token = token.strip()
    if token.isdigit():
        for n in (f'task-{token}', f'task_{token}'):
            p = tasks_root / n
            if p.is_dir():
                return p
        return None
    p = tasks_root / token
    return p if p.is_dir() else None


def _process_single_task(task_dir: Path, *, from_root: bool, dry_run: bool) -> int:
    name = task_dir.name
    try:
        md_path = find_task_markdown(name)
    except FileNotFoundError as e:
        print(f'Warning: {e}', file=sys.stderr)
        return 0
    try:
        ref_files = collect_task_reference_files(name)
    except FileNotFoundError:
        ref_files = []
    # Try references-section aware update; if not present, fall back to global link replacement
    try:
        new_text, notes = update_references_in_md(
            md_path,
            ref_files,
            as_plain=True,  # prefer plain path output in task mode
            from_root=from_root,
            strict=True,
        )
    except RuntimeError as e:
        if 'References' in str(e):
            new_text, notes = update_links_globally(
                md_path,
                ref_files,
                as_plain=True,
                from_root=from_root,
            )
        else:
            raise

    if dry_run:
        print(f'Would update: {md_path}')
        for n in notes:
            print(f'Note: {n}', file=sys.stderr)
        return 0

    md_path.write_text(new_text, encoding='utf-8')
    print(f'Updated {md_path}')
    for n in notes:
        print(f'Note: {n}', file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Update References links in Markdown to local PDF paths (task or case mode). Supports single, multiple, or all tasks.')
    parser.add_argument('name', nargs='?', help="Task token(s) like 'task-3', '3,5' or 'all'; or legacy case name like 'case1'")
    parser.add_argument('--tasks', help="Alternative to positional name: 'all' or CSV like '1,3,5' or 'task-1,task_2'")
    parser.add_argument('--mode', choices=['task', 'case'], help='Force mode detection (default: auto)')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes; just report')
    parser.add_argument('--plain', action='store_true', help='Replace link markup with plain path text (case mode only)')
    parser.add_argument('--from-root', action='store_true', help="Use 'references/...' style paths instead of relative paths")
    args = parser.parse_args(argv)

    # If tasks are requested (via --tasks or positional looks like list/all), run in tasks mode
    task_spec = args.tasks
    if task_spec is None and args.name:
        if args.name.strip().lower() == 'all' or ',' in args.name or args.name.strip().isdigit() or args.name.startswith(('task-', 'task_')):
            task_spec = args.name

    if task_spec is not None:
        tasks_root = _tasks_root()
        if task_spec.strip().lower() == 'all':
            task_dirs = list(_iter_all_tasks(tasks_root))
        else:
            tokens = [t.strip() for t in task_spec.split(',') if t.strip()]
            task_dirs = []
            for t in tokens:
                p = _resolve_task_dir_any(tasks_root, t)
                if p is None:
                    print(f'Warning: task not found for token: {t}', file=sys.stderr)
                    continue
                task_dirs.append(p)
        if not task_dirs:
            print('No matching tasks found.', file=sys.stderr)
            return 1
        rc = 0
        for td in task_dirs:
            rc = rc or _process_single_task(td, from_root=args.from_root, dry_run=args.dry_run)
        return rc

    # Legacy modes (single task or case) using positional 'name'
    if not args.name:
        print("Error: provide a task token, '--tasks', or a case name.", file=sys.stderr)
        return 2

    # Auto-detect mode for single positional
    mode = args.mode
    if mode is None:
        if args.name.startswith(('task-', 'task_')) or (Path('run-and-grade') / 'tasks' / args.name).exists():
            mode = 'task'
        else:
            mode = 'case'

    try:
        if mode == 'task':
            return _process_single_task(_resolve_task_dir_any(_tasks_root(), args.name) or (_tasks_root() / args.name), from_root=args.from_root, dry_run=args.dry_run)
        else:
            md_path = find_case_markdown(args.name)
            ref_files = collect_reference_files(args.name)
            new_text, notes = update_references_in_md(
                md_path,
                ref_files,
                as_plain=args.plain,
                from_root=args.from_root,
                strict=False,
            )
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return 2

    if args.dry_run:
        print(f'Would update: {md_path}')
        for n in notes:
            print(f'Note: {n}', file=sys.stderr)
        return 0

    md_path.write_text(new_text, encoding='utf-8')
    print(f'Updated {md_path}')
    for n in notes:
        print(f'Note: {n}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
