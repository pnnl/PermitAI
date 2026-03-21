from __future__ import annotations

import argparse
import os
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, Iterable, List, Tuple, Optional

from openai import OpenAI, AzureOpenAI
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account as sa
import vertexai 
from vertexai.generative_models import GenerativeModel 
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ---------- Grading core ----------

def grade_with_openai(task_output: str, rubric: str, *, model: str = "gpt-5") -> str:
    """Call OpenAI Responses API to grade a task.

    Returns a JSON string that conforms to the requested schema.
    """
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": rubric},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": task_output},
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "evaluation_table",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "criteria": {
                            "type": "array",
                            "description": "Each element details a criterion, its score (1–5), and the justification.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "criterion": {
                                        "type": "string",
                                        "description": "The name of the evaluation criterion. Use the exact name of the criterion from the rubric, do not add any numbers (i.e. 1., 2., 3., etc.) to the name",
                                    },
                                    "score": {
                                        "type": "integer",
                                        "description": "Score on a 1-5 scale (integer only).",
                                        "minimum": 1,
                                        "maximum": 5,
                                    },
                                    "justification": {
                                        "type": "string",
                                        "description": "Textual justification for the assigned score.",
                                        "minLength": 1,
                                    },
                                },
                                "required": ["criterion", "score", "justification"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["criteria"],
                    "additionalProperties": False,
                },
            },
            "verbosity": "medium",
        },
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
    )
    return response.output_text


def grade_with_azure(
    task_output: str,
    rubric: str,
    *,
    model: str = "gpt-5",
) -> str:
    """
    Call Azure OpenAI Chat Completions API and preserve the JSON schema
    by using a function (tool) whose parameters equal the schema.
    Returns a JSON string conforming to the requested schema.
    """
    endpoint = os.environ['AZURE_ENDPOINT']


    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=endpoint
    )

    schema = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "description": "Each element details a criterion, its score (1–5), and the justification.",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {
                            "type": "string",
                            "description": "The name of the evaluation criterion. Use the exact name of the criterion from the rubric, do not add any numbers (i.e. 1., 2., 3., etc.) to the name",
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score on a 1-5 scale (integer only).",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "justification": {
                            "type": "string",
                            "description": "Textual justification for the assigned score.",
                            "minLength": 1,
                        },
                    },
                    "required": ["criterion", "score", "justification"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["criteria"],
        "additionalProperties": False,
    }

    system_msg = (
        "You are an evaluator. Read the rubric and assess the user's task output. "
        "Return only the function call with arguments that satisfy the parameters schema."
    )

    user_msg = f"Rubric:\n{rubric}\n\nTask Output:\n{task_output}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "submit_evaluation",
                    "description": "Return the structured evaluation table.",
                    "parameters": schema,  # Your JSON Schema enforced here
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
        reasoning_effort="medium"
    )

    tool_calls = resp.choices[0].message.tool_calls
    if not tool_calls:
        raise ValueError("Model did not return a function call.")

    args_json_str = tool_calls[0].function.arguments
    return args_json_str


def grade_with_gemini(
    task_output: str,
    rubric: str,
    *,
    model: str = "gemini-2.5-pro",
) -> str:
    """
    Call Google Vertex AI Gemini to grade a task.

    Returns a JSON string that conforms to the requested schema.
    Uses credentials from env var `VERTEXAI_CREDENTIALS_JSON_PATH` or a default path.
    """

    credentials_path = os.environ.get(
        "VERTEXAI_CREDENTIALS_JSON_PATH",
        "policyai-vertex-creds.json",
    )
    try:
        credentials = sa.Credentials.from_service_account_file(credentials_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Vertex credentials from {credentials_path}: {e}")

    try:
        vertexai.init(project=credentials.project_id, credentials=credentials)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

    model_name = model or "gemini-2.5-pro"
    model = GenerativeModel(model_name)

    # Prompt that enforces the JSON structure.
    schema = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "justification": {"type": "string", "minLength": 1},
                    },
                    "required": ["criterion", "score", "justification"],
                    "additionalProperties": False,
                },
                "description": "Each element details a criterion, its score (1–5), and the justification.",
            }
        },
        "required": ["criteria"],
        "additionalProperties": False,
    }

    prompt = (
        "You are an evaluator. Read the rubric and assess the user's task output.\n"
        "Return ONLY a JSON object that strictly matches the schema below. Do not include extra text.\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Rules:\n"
        "- Use exact criterion names from the rubric; do not add numbering.\n"
        "- Scores must be integers from 1 to 5.\n"
        "- Provide concise, specific justifications.\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Task Output:\n{task_output}"
    )

    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                # "temperature": 0.1,
                # Encourage JSON-only responses
                "response_mime_type": "application/json",
            },
        )
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}")

    # Prefer resp.text; fallback to parsing parts if needed.
    try:
        return resp.text
    except Exception:
        try:
            # Some SDK versions return candidates/parts structures
            cand = getattr(resp, "candidates", None)
            if cand and len(cand) > 0:
                parts = getattr(cand[0].content, "parts", None) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        return t
        except Exception:
            pass
    raise RuntimeError("Gemini response did not contain text output")


def grade_with_claude( task_output: str,
    rubric: str,
    *,
    model: str = "sonnet"):

    session_kwargs = {
        "region_name": os.environ.get("AWS_REGION"),
        "profile_name": os.environ.get("AWS_PROFILE") 
    }
    session_kwargs['aws_access_key_id'] = os.environ.get("AWS_ACCESS_KEY_ID", None)
    session_kwargs['aws_secret_access_key'] = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
    session_kwargs['aws_session_token'] = os.environ.get("AWS_SESSION_TOKEN", None)

    session = boto3.Session(**session_kwargs)

    # Initialize the Bedrock Runtime client
    bedrock_runtime = session.client(service_name='bedrock-runtime', region_name=session_kwargs['region_name']) 

    if model == "sonnet":
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    



    # Define your JSON schema for evaluation
    schema = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "description": "Each element details a criterion, its score (1–5), and the justification.",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {
                            "type": "string",
                            "description": "The name of the evaluation criterion. Use the exact name of the criterion from the rubric, do not add any numbers (i.e. 1., 2., 3., etc.) to the name"
                        },
                        "score": {
                            "type": "integer",
                            "description": "Score on a 1-5 scale (integer only).",
                            "minimum": 1,
                            "maximum": 5
                        },
                        "justification": {
                            "type": "string",
                            "description": "Textual justification for the assigned score.",
                            "minLength": 1
                        }
                    },
                    "required": ["criterion", "score", "justification"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["criteria"],
        "additionalProperties": False
    }



    user_msg = f"You are an evaluator. Read the rubric and assess the user's task output. Return only a JSON object that conforms to the provided schema. Rubric:\n{rubric}\n\nTask Output:\n{task_output}"

   
    # Prepare the payload with JSON schema enforcement
    
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 25000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_msg}
                ]
            }
        ],
        "tools": [
            {
                "name": "grade_draft",
                "description": "Grade draft based on rubric.",
                "input_schema": schema
            }
        ],
        "tool_choice": {"type": "tool", "name": "grade_draft"}
    }


    response = bedrock_runtime.invoke_model(
        modelId=model_id,  
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

        # Parse the response: extract tool_use input payload
    response_body = json.loads(response["body"].read())
    content_blocks = response_body.get("content", []) or []

    tool_payload = None
    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") == "grade_draft":
            tool_payload = block.get("input")
            break

    if isinstance(tool_payload, str):
        # Sometimes models may wrap JSON as a string; attempt to parse.
        try:
            tool_payload = json.loads(tool_payload)
        except Exception:
            raise ValueError("Claude tool_use input was a string that could not be parsed as JSON.")

    if not isinstance(tool_payload, dict):
        raise ValueError("Claude did not return a tool_use block with structured input.")

    # Normalize structure to match expected schema: criteria must be list[dict]
    criteria = tool_payload.get("criteria", [])
    if isinstance(criteria, str):
        try:
            criteria = json.loads(criteria)
        except Exception:
            criteria = []
    if not isinstance(criteria, list):
        criteria = []

    normalized: list[dict] = []
    for item in criteria:
        if isinstance(item, dict):
            crit = str(item.get("criterion", "")).strip()
            just = str(item.get("justification", "")).strip()
            sc = item.get("score", 0)
            try:
                sc_int = int(sc)
            except Exception:
                try:
                    sc_int = int(float(sc))
                except Exception:
                    sc_int = 0
            normalized.append({
                "criterion": crit,
                "score": sc_int,
                "justification": just,
            })
        # Ignore non-dict items to prevent downstream crashes

    tool_payload["criteria"] = normalized
    return json.dumps(tool_payload)



def summarize_grade(grade_data: Dict[str, Any], wrap_width: int = 100) -> Tuple[int, str]:
    """
    Accepts a dict in the shape of json.loads(grade).
    Returns (total_score, formatted_summary).
    """
    criteria = grade_data.get("criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("grade_data['criteria'] must be a list")

    total = 0
    max_per_criterion = 5
    lines = []

    for idx, item in enumerate(criteria, 1):
        name = str(item.get("criterion", "")).strip()
        score = int(item.get("score", 0) or 0)
        justification = str(item.get("justification", "")).strip()
        total += score

        header = f"{idx}) {name} — {score}/{max_per_criterion}"
        lines.append(header)
        if justification:
            wrapped = fill(
                justification,
                width=wrap_width,
                initial_indent="   ",
                subsequent_indent="   "
            )
            lines.append(wrapped)
        lines.append("")

    count = len(criteria)
    max_total = max_per_criterion * count if count else 0
    average = (total / count) if count else 0.0

    summary_header = (
        "Evaluation Summary\n"
        f"Total: {total} / {max_total} | Average: {average:.2f} / {max_per_criterion} | Criteria: {count}\n"
    )

    formatted = summary_header + "\n".join(lines).rstrip()
    return total, formatted

def save_grade_report(grade_data: Dict[str, Any], output_path: str, wrap_width: int = 100) -> Tuple[int, str]:
    """
    Creates the formatted scorecard and saves it to output_path.
    Returns (total_score, output_path).
    """
    total, report = summarize_grade(grade_data, wrap_width=wrap_width)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")
    return total, str(path)


# ---------- Aggregate summary/report ----------

def build_aggregate_summary(combined: Dict[str, Any], wrap_width: int = 100) -> str:
    name = combined.get("task", "task")
    model = combined.get("model", "")
    n_trials = int(combined.get("n_trials", 0))
    rubric_path = combined.get("rubric_path", "")
    report_path = combined.get("report_path", "")
    stats = combined.get("stats", {}) or {}
    criteria_stats = stats.get("criteria", []) or []
    final_stats = stats.get("final", {}) or {}

    header = (
        "Evaluation Summary (Aggregated)\n"
        f"Task: {name} | Model: {model} | Trials: {n_trials}\n"
        f"Rubric: {rubric_path}\n"
        f"Report: {report_path}\n\n"
    )

    lines: List[str] = [header]

    for idx, c in enumerate(criteria_stats, 1):
        crit_name = str(c.get("criterion", "")).strip()
        avg = float(c.get("average", 0.0))
        std = float(c.get("stddev", 0.0))
        cmin = c.get("min", 0)
        cmax = c.get("max", 0)
        n = int(c.get("n", 0))
        line = f"{idx}) {crit_name} — avg={avg:.2f}, std={std:.2f}, min={cmin}, max={cmax}, n={n}"
        lines.append(line)

    if final_stats:
        favg = float(final_stats.get("average", 0.0))
        fstd = float(final_stats.get("stddev", 0.0))
        fmin = float(final_stats.get("min", 0.0))
        fmax = float(final_stats.get("max", 0.0))
        fn = int(final_stats.get("n", 0))
        lines.append("")
        lines.append(f"Overall — avg={favg:.2f}, std={fstd:.2f}, min={fmin:.2f}, max={fmax:.2f}, n={fn}")

    return "\n".join(lines).rstrip() + "\n"


def save_aggregate_report(combined: Dict[str, Any], output_path: str, wrap_width: int = 100) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_aggregate_summary(combined, wrap_width=wrap_width)
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------- CLI + Task orchestration ----------

@dataclass
class TaskPaths:
    name: str
    root: Path
    workdir: Path
    report_md: Path
    rubric_md: Path


def find_report_md_path(base: Path) -> Path | None:
    """Find the first 'report.md' under 'base' (recursively).
    Preference order: shortest relative path depth, then lexicographic.
    Returns None if not found.
    """
    try:
        candidates = list(base.rglob("report.md")) if base.exists() else []
    except Exception:
        candidates = []
    if not candidates:
        return None
    # Sort by depth (number of parts) then lexicographically for determinism
    candidates.sort(key=lambda p: (len(p.relative_to(base).parts), str(p)))
    return candidates[0]


def find_agent_report_path(results_root: Path, agent: str, task_name: str) -> Path | None:
    """Backward-compatible single-report finder.
    Looks for results/<agent>/<task>/out/report.md or within out/ recursively.
    """
    base = results_root / agent / task_name / "out"
    exact = base / "report.md"
    if exact.is_file():
        return exact
    try:
        for p in base.rglob("report.md"):
            return p
    except Exception:
        pass
    return None


def find_agent_trial_report_paths(results_root: Path, agent: str, task_name: str) -> List[Tuple[str, Path]]:
    """Return a list of (trial_tag, report_path) for all trials of a task.
    New layout:
      results/<agent>/<task>/trial-*/out/report.md
    Falls back to a single 'out/report.md' as one trial when no trial dirs exist.
    """
    task_base = results_root / agent / task_name
    trials: List[Tuple[str, Path]] = []
    try:
        trial_dirs = [d for d in task_base.iterdir() if d.is_dir() and d.name.startswith("trial-")]
    except Exception:
        trial_dirs = []

    for tdir in sorted(trial_dirs, key=lambda p: p.name):
        out_base = tdir / "out"
        exact = out_base / "report.md"
        if exact.is_file():
            trials.append((tdir.name, exact))
            continue
        # Fallback recursive search within out/
        try:
            for p in out_base.rglob("report.md"):
                trials.append((tdir.name, p))
                break
        except Exception:
            pass

    if trials:
        return trials

    # Fallback: treat legacy single out/ as one trial
    legacy = find_agent_report_path(results_root, agent, task_name)
    if legacy is not None:
        return [("trial-1", legacy)]
    return []


def detect_tasks_root(base: Path) -> Path:
    """Detect the tasks root directory.
    Tries common layouts:
      - <cwd>/run-and-grade/tasks
      - <cwd>/run_and_grade/tasks
      - <cwd>/tasks (when running from inside run-and-grade)
    Returns the first existing directory containing task-* subdirs; otherwise
    falls back to <cwd>/run-and-grade/tasks.
    """
    candidates = [
        base / "run-and-grade" / "tasks",
        base / "run_and_grade" / "tasks",
        base / "tasks",
    ]
    for c in candidates:
        try:
            if c.is_dir() and any(d.is_dir() and d.name.startswith("task-") for d in c.iterdir()):
                return c
        except Exception:
            pass
    # Fallback default
    return base / "run-and-grade" / "tasks"


def discover_all_tasks(base: Path) -> List[TaskPaths]:
    tasks: List[TaskPaths] = []
    root = detect_tasks_root(base)
    # Detect evaluation documents root (where rubrics live now)
    def _detect_eval_docs_root(base: Path) -> Path:
        candidates = [
            base / "run-and-grade" / "evaluation-documents",
            base / "run_and_grade" / "evaluation-documents",
            base / "evaluation-documents",
        ]
        for c in candidates:
            try:
                if c.is_dir() and any(d.is_dir() and d.name.startswith("task-") for d in c.iterdir()):
                    return c
            except Exception:
                pass
        return candidates[0]

    eval_root = _detect_eval_docs_root(base)
    for entry in sorted(root.glob("task-*"), key=lambda p: int(re.sub(r"^task-", "", p.name)) if re.match(r"^task-\d+$", p.name) else float("inf")):
        if not entry.is_dir() or not re.match(r"^task-\d+$", entry.name):
            continue
        workdir = entry / "workdir"
        run_base = workdir if workdir.is_dir() else entry
        report_md = find_report_md_path(run_base) or (run_base / "report.md")
        # Rubric now resides under evaluation-documents/<task>/rubric.md
        rubric_md = eval_root / entry.name / "rubric.md"
        tasks.append(TaskPaths(name=entry.name, root=entry, workdir=run_base, report_md=report_md, rubric_md=rubric_md))
    return tasks


def select_tasks(base: Path, arg: str) -> List[TaskPaths]:
    arg_lc = arg.strip().lower()
    if arg_lc == "all":
        return discover_all_tasks(base)

    # Support single number or comma-separated list for robustness
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    selected = []
    root = detect_tasks_root(base)
    # Detect evaluation documents root (where rubrics live now)
    def _detect_eval_docs_root(base: Path) -> Path:
        candidates = [
            base / "run-and-grade" / "evaluation-documents",
            base / "run_and_grade" / "evaluation-documents",
            base / "evaluation-documents",
        ]
        for c in candidates:
            try:
                if c.is_dir() and any(d.is_dir() and d.name.startswith("task-") for d in c.iterdir()):
                    return c
            except Exception:
                pass
        return candidates[0]

    eval_root = _detect_eval_docs_root(base)

    for p in parts:
        if not p.isdigit():
            raise SystemExit(f"Invalid task specifier: '{p}'. Use 'all' or numbers like '1' or '1,2'.")
        d = root / f"task-{int(p)}"
        if not d.is_dir():
            raise SystemExit(f"Task not found: {d}")
        workdir = d / "workdir"
        run_base = workdir if workdir.is_dir() else d
        report_md = find_report_md_path(run_base) or (run_base / "report.md")
        selected.append(
            TaskPaths(
                name=d.name,
                root=d,
                workdir=run_base,
                report_md=report_md,
                # Rubric now resides under evaluation-documents/<task>/rubric.md
                rubric_md=eval_root / d.name / "rubric.md",
            )
        )
    return selected


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Missing required file: {path}")
    except Exception as e:
        raise SystemExit(f"Failed to read {path}: {e}")


def save_raw_json(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute_trial_overall(criteria: List[Dict[str, Any]]) -> float:
    # Try to use a criterion named like "Final Score"
    final_candidates = [c for c in criteria if "final" in str(c.get("criterion", "")).lower() and "score" in c]
    if final_candidates:
        try:
            return float(final_candidates[0].get("score", 0))
        except Exception:
            pass
    # Fallback: mean of all criterion scores excluding those that look like final
    scores = [int(c.get("score", 0)) for c in criteria if "final" not in str(c.get("criterion", "")).lower()]
    return float(sum(scores) / len(scores)) if scores else 0.0


def aggregate_stats(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Map normalized criterion name -> (display_name, list of scores)
    buckets: Dict[str, Dict[str, Any]] = {}
    trial_overalls: List[float] = []

    for t in trials:
        crits = t.get("criteria", []) if isinstance(t, dict) else []
        trial_overalls.append(compute_trial_overall(crits))
        for c in crits:
            name_raw = str(c.get("criterion", "")).strip()
            if not name_raw:
                continue
            key = name_raw.lower()
            score = int(c.get("score", 0) or 0)
            b = buckets.setdefault(key, {"display": name_raw, "scores": []})
            # Preserve first seen display name
            if not b.get("display"):
                b["display"] = name_raw
            b["scores"].append(score)

    # Build stats list
    stats_criteria: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        disp = buckets[key]["display"]
        scores = buckets[key]["scores"]
        n = len(scores)
        avg = float(sum(scores) / n) if n else 0.0
        if n > 1:
            std = float(statistics.stdev(scores))
        else:
            std = 0.0
        min_v = int(min(scores)) if n else 0
        max_v = int(max(scores)) if n else 0
        stats_criteria.append({
            "criterion": disp,
            "average": avg,
            "stddev": std,
            "min": min_v,
            "max": max_v,
            "n": n,
        })

    # Final overall
    n_overall = len(trial_overalls)
    avg_overall = float(sum(trial_overalls) / n_overall) if n_overall else 0.0
    std_overall = float(statistics.stdev(trial_overalls)) if n_overall > 1 else 0.0
    min_overall = float(min(trial_overalls)) if n_overall else 0.0
    max_overall = float(max(trial_overalls)) if n_overall else 0.0

    return {
        "criteria": stats_criteria,
        "final": {"average": avg_overall, "stddev": std_overall, "min": min_overall, "max": max_overall, "n": n_overall},
        "trial_overalls": trial_overalls,
    }


def run_for_task(tp: TaskPaths, *, model: str = "gpt-5", results_root: Path, agent: str, trials: int = 1, workers: Optional[int] = None) -> Tuple[str, int, str | None]:
    """
    Grade all execution trials for a single task.
    For each trial found under results/<agent>/<task>/trial-*/, generate a separate JSON result.
    Also writes a combined index JSON listing per-trial outputs.

    Returns tuple: (task_name, exit_code, combined_json_path)
    """
    if not tp.rubric_md.is_file():
        print(f"[skip] {tp.name}: missing rubric at {tp.rubric_md}", file=sys.stderr)
        return tp.name, 2, None

    # Discover all trial report paths
    trial_reports = find_agent_trial_report_paths(results_root, agent, tp.name)
    if not trial_reports:
        looked_base = results_root / agent / tp.name
        print(f"[skip] {tp.name}: no trial outputs found under {looked_base}", file=sys.stderr)
        return tp.name, 2, None

    rubric = read_text_file(tp.rubric_md).strip()
    if not rubric:
        print(f"[skip] {tp.name}: rubric is empty at {tp.rubric_md}", file=sys.stderr)
        return tp.name, 2, None

    grades_dir = results_root / agent / tp.name / "grades" / model
    grades_dir.mkdir(parents=True, exist_ok=True)

    per_trial_results: List[Dict[str, Any]] = []
    mlc = (model or "").lower()
    if "gemini" in mlc:
        grade_fn = grade_with_gemini
    elif "sonnet" in mlc:
        grade_fn = grade_with_claude
    else:
        grade_fn = grade_with_azure

    for trial_tag, report_path in trial_reports:
        # If a grade for this trial already exists for this model, reuse it and skip re-grading
        trial_json_path = grades_dir / f"{trial_tag}-grade.json"
        if trial_json_path.is_file():
            try:
                existing = json.loads(trial_json_path.read_text(encoding="utf-8"))
                n_evals = int(existing.get("n_evaluations", 0))
                rep_path = str(existing.get("report_path", str(report_path)))
                per_trial_results.append({
                    "trial": trial_tag,
                    "report_path": rep_path,
                    "grade_json": str(trial_json_path),
                    "n_evaluations": n_evals,
                })
                print(f"[skip] {tp.name}/{trial_tag}: existing grade found at {trial_json_path}")
                continue
            except Exception:
                # Fall through to re-grade if the existing JSON is unreadable
                print(f"[warn] {tp.name}/{trial_tag}: existing grade unreadable, re-grading")

        task_output = read_text_file(report_path).strip()
        if not task_output:
            print(f"[skip] {tp.name}/{trial_tag}: report is empty at {report_path}", file=sys.stderr)
            continue

        # Run grading repeats for this trial in parallel (default: 1)
        parsed_evals: List[Dict[str, Any]] = []
        errors: List[str] = []
        max_workers = workers or trials or 1
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(grade_fn, task_output, rubric, model=model) for _ in range(max(trials, 1))]
            for fut in as_completed(futs):
                try:
                    s = fut.result()
                    parsed_evals.append(json.loads(s))
                except Exception as e:
                    errors.append(str(e))

        if not parsed_evals:
            print(f"[fail] {tp.name}/{trial_tag}: grading failed. Example error: {errors[0] if errors else 'unknown'}", file=sys.stderr)
            continue

        # Save per-trial JSON
        trial_payload = {
            "task": tp.name,
            "trial": trial_tag,
            "model": model,
            "rubric_path": str(tp.rubric_md),
            "report_path": str(report_path),
            "n_evaluations": len(parsed_evals),
            "evaluations": parsed_evals,
            "stats": aggregate_stats(parsed_evals),
        }
        save_raw_json(trial_payload, trial_json_path)
        print(f"- {tp.name}/{trial_tag}: saved JSON to {trial_json_path}")

        per_trial_results.append({
            "trial": trial_tag,
            "report_path": str(report_path),
            "grade_json": str(trial_json_path),
            "n_evaluations": len(parsed_evals),
        })

    if not per_trial_results:
        return tp.name, 1, None

    # Write combined index JSON listing all per-trial results
    combined_json_path = grades_dir / f"{tp.name}-grade.json"
    combined = {
        "task": tp.name,
        "model": model,
        "rubric_path": str(tp.rubric_md),
        "n_trials": len(per_trial_results),
        "trial_results": per_trial_results,
    }
    save_raw_json(combined, combined_json_path)
    print(f"- {tp.name}: wrote combined index to {combined_json_path}")
    return tp.name, 0, str(combined_json_path)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grade task outputs against rubrics using Azure OpenAI.")
    p.add_argument("--task", required=True, help="'all' or a number like '1' or '1,2'")
    p.add_argument("--model", default="gpt-5", help="Model name (default: gpt-5)")
    p.add_argument(
        "--results-root",
        default=None,
        help="Base results directory containing agent folders (auto-detects run-and-grade/results, run_and_grade/results, or results/)",
    )
    p.add_argument("--agent", default=None, help="Agent name under results/ to grade (e.g., 'codex')")
    p.add_argument("--trials", type=int, default=1, help="Number of grading evaluations to run in parallel per execution trial")
    p.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: trials)")
    return p.parse_args(list(argv))


def _detect_results_root(base: Path, provided: Optional[str]) -> Path:
    if provided:
        return Path(provided)
    candidates = [
        base / "run-and-grade" / "results",
        base / "run_and_grade" / "results",
        base / "results",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fallback to first candidate even if it doesn't exist
    return candidates[0]


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    base = Path.cwd()
    try:
        tasks = select_tasks(base, args.task)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    if not tasks:
        print("No tasks to process.")
        return 0

    results_root = _detect_results_root(base, args.results_root)

    print(results_root)

    # Determine agent name
    agent = args.agent
    if not agent:
        # Try to auto-detect a single agent directory under results_root
        try:
            agent_dirs = [d.name for d in results_root.iterdir() if d.is_dir()]
        except Exception:
            agent_dirs = []
        if len(agent_dirs) == 1:
            agent = agent_dirs[0]
            print(f"Auto-detected agent: {agent}")
        else:
            print("Error: please specify --agent (found: {} under {}).".format(
                ", ".join(agent_dirs) if agent_dirs else "none", results_root
            ), file=sys.stderr)
            return 2

    results = []
    for tp in tasks:
        print(f"\n=== Grading {tp.name} (trials={args.trials}) ===")
        results.append(run_for_task(tp, model=args.model, results_root=results_root, agent=agent, trials=max(args.trials, 1), workers=args.workers))

    print("\n=== Summary ===")
    rc = 0
    for name, code, rawpath in results:
        if code == 0:
            print(f"- {name}: ok ({rawpath})")
        else:
            print(f"- {name}: fail (rc={code})")
            rc = max(rc, code)

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
