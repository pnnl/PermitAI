#!/usr/bin/env python3
"""
DOCX -> Markdown converter (no external deps).

Features supported (basic):
- Headings via paragraph style (Heading1..Heading6)
- Paragraphs and line breaks
- Bulleted/numbered lists (detected via w:numPr)
- Bold, italic, bold+italic
- Hyperlinks (resolves r:id via word/_rels/document.xml.rels)

Also includes a --make-sample option to generate a minimal sample.docx
and convert it to sample.md for quick verification.
"""
import argparse
import sys
import zipfile
from xml.etree import ElementTree as ET
from pathlib import Path

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NS = {
    'w': W_NS,
    'r': R_NS,
}


def _read_xml_from_zip(zf: zipfile.ZipFile, path: str):
    with zf.open(path) as f:
        data = f.read()
    return ET.fromstring(data)


def _load_relationships(zf: zipfile.ZipFile):
    rel_path = 'word/_rels/document.xml.rels'
    rels = {}
    try:
        root = _read_xml_from_zip(zf, rel_path)
    except KeyError:
        return rels
    for rel in root.findall(f'.//{{{PKG_REL_NS}}}Relationship'):
        rid = rel.get('Id')
        target = rel.get('Target')
        rtype = rel.get('Type')
        rels[rid] = {'Target': target, 'Type': rtype}
    return rels


def _run_text(run: ET.Element) -> str:
    # Concatenate all w:t inside this run respecting xml:space
    texts = []
    for t in run.findall(f'.//w:t', NS):
        txt = t.text or ''
        # Handle preserved leading/trailing spaces if present
        if t.get('{http://www.w3.org/XML/1998/namespace}space') == 'preserve':
            texts.append(txt)
        else:
            texts.append(txt)
    return ''.join(texts)


def _format_run(run: ET.Element, text: str) -> str:
    rpr = run.find('w:rPr', NS)
    if rpr is None or not text:
        return text
    is_bold = rpr.find('w:b', NS) is not None
    is_italic = rpr.find('w:i', NS) is not None
    if is_bold and is_italic:
        return f'***{text}***'
    if is_bold:
        return f'**{text}**'
    if is_italic:
        return f'*{text}*'
    return text


def _paragraph_is_list(p: ET.Element) -> bool:
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        return False
    return ppr.find('w:numPr', NS) is not None


def _heading_level(p: ET.Element):
    ppr = p.find('w:pPr', NS)
    if ppr is None:
        return None
    pstyle = ppr.find('w:pStyle', NS)
    if pstyle is None:
        return None
    val = pstyle.get(f'{{{W_NS}}}val') or pstyle.get('w:val')
    if not val:
        return None
    if val.lower().startswith('heading'):
        try:
            lvl = int(''.join(ch for ch in val if ch.isdigit()))
            return max(1, min(6, lvl))
        except Exception:
            return 1
    return None


def _collect_inline_children(p: ET.Element, rels: dict) -> str:
    parts = []
    for child in p:
        tag = child.tag
        if tag == f'{{{W_NS}}}r':
            raw = _run_text(child)
            parts.append(_format_run(child, raw))
        elif tag == f'{{{W_NS}}}hyperlink':
            rid = child.get(f'{{{R_NS}}}id')
            url = None
            if rid and rid in rels:
                url = rels[rid].get('Target')
            # gather inner runs text with formatting
            inner = []
            for r in child.findall('w:r', NS):
                raw = _run_text(r)
                inner.append(_format_run(r, raw))
            label = ''.join(inner)
            if url:
                parts.append(f'[{label}]({url})')
            else:
                parts.append(label)
        elif tag == f'{{{W_NS}}}tab':
            parts.append('\t')
        elif tag == f'{{{W_NS}}}br':
            parts.append('  \n')
        # Ignore other tags (bookmarkStart, proofErr, etc.)
    return ''.join(parts)


def _cell_text(tc: ET.Element, rels: dict) -> str:
    """Extract concatenated text from a table cell (w:tc), joining paragraphs with <br>."""
    paras = tc.findall('.//w:p', NS)
    texts = []
    for p in paras:
        txt = _collect_inline_children(p, rels).strip()
        if txt:
            texts.append(txt)
    return '<br>'.join(texts)


def _table_to_markdown(tbl: ET.Element, rels: dict) -> str:
    """Convert a DOCX table (w:tbl) to GitHub-flavored Markdown table.
    - Uses first row as header.
    - Supports gridSpan by padding empty cells; vMerge continuations become empty.
    """
    rows_data = []
    max_cols = 0
    for tr in tbl.findall('w:tr', NS):
        row_cells = []
        for tc in tr.findall('w:tc', NS):
            tcpr = tc.find('w:tcPr', NS)
            # Handle vertical merge continuation as empty cell
            if tcpr is not None:
                vmerge = tcpr.find('w:vMerge', NS)
                if vmerge is not None and vmerge.get(f'{{{W_NS}}}val') is None:
                    # continuation of vMerge; represent as empty cell
                    txt = ''
                else:
                    txt = _cell_text(tc, rels)
            else:
                txt = _cell_text(tc, rels)

            # gridSpan handling
            span = 1
            if tcpr is not None:
                g = tcpr.find('w:gridSpan', NS)
                try:
                    span = int(g.get(f'{{{W_NS}}}val')) if g is not None else 1
                except Exception:
                    span = 1
            # Append cell text and pad for span-1
            row_cells.append(txt)
            for _ in range(max(span - 1, 0)):
                row_cells.append('')
        max_cols = max(max_cols, len(row_cells))
        rows_data.append(row_cells)

    if not rows_data:
        return ''

    # Normalize row lengths
    norm_rows = [cells + ([''] * (max_cols - len(cells))) for cells in rows_data]

    # Build Markdown
    lines = []
    header = norm_rows[0] if norm_rows else [''] * max_cols
    sep = ['---'] * max_cols
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(sep) + ' |')
    for r in norm_rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def docx_to_markdown(input_path: Path) -> str:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    with zipfile.ZipFile(input_path, 'r') as zf:
        doc = _read_xml_from_zip(zf, 'word/document.xml')
        rels = _load_relationships(zf)

    body = doc.find('w:body', NS)
    if body is None:
        return ''

    blocks = []  # each block is a multi-line string; we'll separate with blank lines
    # Iterate over direct children to preserve order of paragraphs and tables
    for child in list(body):
        tag = child.tag
        if tag == f'{{{W_NS}}}p':
            text = _collect_inline_children(child, rels).strip()
            if text == '':
                continue
            level = _heading_level(child)
            if level:
                blocks.append(f"{'#' * level} {text}")
                continue
            if _paragraph_is_list(child):
                # accumulate list items into a single block
                if blocks and blocks[-1].startswith('- '):
                    # append to previous list block
                    blocks[-1] = blocks[-1] + f"\n- {text}"
                else:
                    blocks.append(f"- {text}")
                continue
            blocks.append(text)
        elif tag == f'{{{W_NS}}}tbl':
            tbl_md = _table_to_markdown(child, rels).strip()
            if tbl_md:
                blocks.append(tbl_md)
        # ignore other tags (sectPr, etc.)

    # Join blocks with single blank line between
    md_text = '\n\n'.join(blocks).rstrip() + '\n'

    return md_text


def write_markdown(md_text: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_text, encoding='utf-8')


def make_sample_docx(output_path: Path):
    """Create a minimal, valid DOCX with headings, styles, list, and link."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '</Types>'
    ).encode('utf-8')

    rels_top = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '</Relationships>'
    ).encode('utf-8')

    # Basic document.xml with heading, paragraph with bold/italic, list, and hyperlink
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"\n'
        '             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
        '  <w:body>\n'
        '    <w:p>\n'
        '      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>\n'
        '      <w:r><w:t>Sample Title</w:t></w:r>\n'
        '    </w:p>\n'
        '    <w:p>\n'
        '      <w:r><w:t>This is a paragraph with </w:t></w:r>\n'
        '      <w:r><w:rPr><w:b/></w:rPr><w:t>bold</w:t></w:r>\n'
        '      <w:r><w:t> and </w:t></w:r>\n'
        '      <w:r><w:rPr><w:i/></w:rPr><w:t>italic</w:t></w:r>\n'
        '      <w:r><w:t> text.</w:t></w:r>\n'
        '    </w:p>\n'
        '    <w:p>\n'
        '      <w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>\n'
        '      <w:r><w:t>First bullet</w:t></w:r>\n'
        '    </w:p>\n'
        '    <w:p>\n'
        '      <w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>\n'
        '      <w:r><w:t>Second bullet</w:t></w:r>\n'
        '    </w:p>\n'
        '    <w:p>\n'
        '      <w:r><w:t>Link: </w:t></w:r>\n'
        '      <w:hyperlink r:id="rId2">\n'
        '        <w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t>OpenAI</w:t></w:r>\n'
        '      </w:hyperlink>\n'
        '    </w:p>\n'
        '    <w:sectPr/>\n'
        '  </w:body>\n'
        '</w:document>'
    ).encode('utf-8')

    # Relationships for hyperlink
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"\n'
        '               xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
        '  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://openai.com" TargetMode="External"/>\n'
        '</Relationships>'
    ).encode('utf-8')

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels_top)
        z.writestr('word/document.xml', document_xml)
        z.writestr('word/_rels/document.xml.rels', doc_rels)


def _tasks_root_from_here() -> Path:
    here = Path(__file__).resolve()
    # repo_root is ../ from utils/
    repo_root = here.parent.parent
    candidate = repo_root / 'run-and-grade' / 'tasks'
    return candidate


def _process_task_dir(task_name: str) -> int:
    """Given a task name (e.g., 'task-3'), find its *task_documents folder and
    convert task.docx -> workdir/task.md and combine rubric.docx + ground_truth.docx -> rubric.md.
    """
    tasks_root = _tasks_root_from_here()
    task_dir = tasks_root / task_name
    if not task_dir.is_dir():
        print(f'Error: task directory not found: {task_dir}', file=sys.stderr)
        return 2

    workdir = task_dir / 'workdir'
    if not workdir.is_dir():
        print(f'Error: missing workdir for task: {workdir}', file=sys.stderr)
        return 2

    # Find the *task_documents directory inside workdir
    doc_dirs = [d for d in workdir.iterdir() if d.is_dir() and d.name.endswith('task_documents')]
    if not doc_dirs:
        print(f'Error: no *task_documents directory found under {workdir}', file=sys.stderr)
        return 2
    if len(doc_dirs) > 1:
        print(f'Warning: multiple *task_documents directories found; using the first: {doc_dirs[0]}', file=sys.stderr)
    doc_dir = doc_dirs[0]

    task_docx = doc_dir / 'task.docx'
    rubric_docx = doc_dir / 'rubric.docx'
    gt_docx = doc_dir / 'ground_truth.docx'

    # Convert task.docx -> workdir/task.md
    if task_docx.is_file():
        try:
            task_md = docx_to_markdown(task_docx)
            out_task_md = workdir / 'task.md'
            write_markdown(task_md, out_task_md)
            print(f'Wrote {out_task_md}')
        except Exception as e:
            print(f'Error converting {task_docx}: {e}', file=sys.stderr)
            return 1
    else:
        print(f'Warning: missing task.docx at {task_docx}', file=sys.stderr)

    # Build rubric.md that preserves rubric content and appends ground truth at the end
    parts = []
    if rubric_docx.is_file():
        try:
            rubric_md = docx_to_markdown(rubric_docx).strip()
            parts.append(rubric_md)
        except Exception as e:
            print(f'Error converting {rubric_docx}: {e}', file=sys.stderr)
            return 1
    else:
        print(f'Warning: missing rubric.docx at {rubric_docx}', file=sys.stderr)

    if gt_docx.is_file():
        try:
            gt_md = docx_to_markdown(gt_docx).strip()
            parts.append('')
            parts.append('# Ground Truth')
            parts.append('')
            parts.append(gt_md)
        except Exception as e:
            print(f'Error converting {gt_docx}: {e}', file=sys.stderr)
            return 1
    else:
        print(f'Warning: missing ground_truth.docx at {gt_docx}', file=sys.stderr)

    if parts:
        combined = '\n'.join(parts).rstrip() + '\n'
        out_rubric_md = task_dir / 'rubric.md'
        try:
            write_markdown(combined, out_rubric_md)
            print(f'Wrote {out_rubric_md}')
        except Exception as e:
            print(f'Error writing rubric.md at {out_rubric_md}: {e}', file=sys.stderr)
            return 1
    else:
        print('Warning: neither rubric.docx nor ground_truth.docx were found; no rubric.md created', file=sys.stderr)

    return 0


def _iter_all_tasks(tasks_root: Path):
    # Yield both dash and underscore styles; prefer dash where both exist
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
    # Accept numeric id
    if token.isdigit():
        for name in (f'task-{token}', f'task_{token}'):
            p = tasks_root / name
            if p.is_dir():
                return p
        return None
    # Accept explicit name
    p = tasks_root / token
    return p if p.is_dir() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert .docx to Markdown, or process task folders.')
    parser.add_argument('input', nargs='?', help='Path to input .docx OR task spec: task name, comma list (e.g., "task-3,task-7" or "3,7"), or "all"')
    parser.add_argument('-o', '--output', help='Output .md path (used only for single .docx input)')
    parser.add_argument('--tasks', help='Alternative to positional input: "all" or comma-separated tasks (ids or names)')
    args = parser.parse_args(argv)

    # Determine operation mode
    in_arg = args.input
    if args.tasks:
        task_spec = args.tasks.strip()
    else:
        task_spec = None

    # Single-file mode (explicit .docx path)
    if in_arg:
        in_path = Path(in_arg)
        if in_path.exists() and in_path.is_file():
            out_path = Path(args.output) if args.output else in_path.with_suffix('.md')
            md_text = docx_to_markdown(in_path)
            write_markdown(md_text, out_path)
            print(f'Wrote {out_path}')
            return 0

    # Task mode detection: via --tasks or positional input being 'all' or a CSV or a single task token
    if task_spec is None and in_arg:
        lowered = in_arg.strip().lower()
        if lowered == 'all' or ',' in in_arg or not Path(in_arg).exists():
            task_spec = in_arg

    if task_spec:
        tasks_root = _tasks_root_from_here()
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
            res = _process_task_dir(td.name)
            rc = rc or res
        return rc

    # If we reach here, we didn't get a valid input
    print('Error: supply a .docx file to convert, a task name, a comma-separated task list, or "all".', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
