# Evaluation Directory

This directory contains evaluator classes for the MAPLE framework that handle different NEPA evaluation tasks using language models.

## Table of Contents

- [Overview](#overview)
- [Base Evaluator Architecture](#base-evaluator-architecture)
- [Derived Evaluator Classes](#derived-evaluator-classes)
  - [NEPA Document Analysis](#nepa-document-analysis)
  - [Public Comment Analysis](#public-comment-analysis)
  - [Geographic Analysis](#geographic-analysis)
- [Creating a New Evaluator](#creating-a-new-evaluator)
- [Common Patterns](#common-patterns)
- [Dependencies](#dependencies)

---

## Overview

The `evaluation/` directory provides specialized evaluators for:

- Question-answering on NEPA documents (multiple context modes)
- Information extraction from NEPA documents (CX, EA, EIS)
- Tribe name extraction for tribal consultation analysis
- Structured extraction from Federal Register notices
- Public comment delineation, binning, classification, and summarization
- Geographic map classification

All evaluators inherit from `BaseEvaluator` and operate on benchmark entry objects from `utils/`.

---

## Base Evaluator Architecture

### `BaseEvaluator` (`base_evaluator.py`)

Abstract base class that all evaluators must implement.

**Constructor**:
```python
BaseEvaluator(
    llm_handler: BaseLLMHandler,
    prompt_template_source: str,
    is_template_file: bool = True
)
```
- `llm_handler`: Any `BaseLLMHandler` instance from `llm_handlers/`
- `prompt_template_source`: Path to a prompt template file, or the template string itself
- `is_template_file`: Set to `False` to pass the template as a raw string instead of a file path

**Attributes**:
- `self.llm_handler` — the LLM handler instance
- `self.prompt_manager` — a `PromptManager` loaded from the template source

### Abstract Methods

Every subclass must implement these three methods:

```python
def prepare_prompt(self, entry, **kwargs) -> Tuple[str, Optional[List[str]]]:
    """Returns (text_prompt, image_paths_or_None)."""

def evaluate_entry(self, entry, **kwargs) -> Optional[Dict[str, Any]]:
    """Runs sequential LLM inference for a single entry. Returns result dict or None."""

def extract_response(self, entry, response: str) -> Optional[Dict[str, Any]]:
    """Parses and validates a raw LLM response string. Used in batch job workflows."""
```

### Batch Processing Methods

**`evaluate_batch`** — sequential evaluation with resume support:
```python
results_path = evaluator.evaluate_batch(
    entries=benchmark_entries,
    output_path="results/task_results.json",
    continue_from_previous=True,   # default: True
    max_attempts=10,
    retry_delay=30
)
```
Processes each entry by calling `evaluate_entry()`, writes results to JSON after every successful entry, and skips already-evaluated entries when `continue_from_previous=True`.

**`execute_gemini_batch_job`** — full GCS-based batch workflow (Google GenAI only):
```python
results_path = evaluator.execute_gemini_batch_job(
    entries=benchmark_entries,
    output_path="results/task_results.json",
    continue_from_previous=True,
    # Batch job routing
    bucket_name="maple_evaluations",
    upload_file_path="input_data/prompts.jsonl",
    predictions_file_path="responses/",
    batch_response_path="results/raw_responses.json",
    prompt_jsonl_path="batch_prompts.jsonl",
    # Image handling (for multimodal tasks)
    upload_images_to_gcs=False,    # True = GCS URI, False = base64
    # Structured output
    structured_response=False      # True = use entry.schema_json for response schema
)
```
Steps internally: identify unevaluated entries → `prepare_prompt()` for each → save JSONL → submit batch job → `extract_response()` for each raw response → save/merge results.

Requires the LLM handler to implement `batch_job()` (currently only `GoogleGenAIHandler`).

### Result Management Methods

| Method | Description |
|---|---|
| `_save_results(results, path)` | Write full results dict to JSON |
| `_load_existing_results(path)` | Load results dict from JSON (returns `{}` if file absent) |
| `_append_result(results, path)` | Overwrite JSON file with updated dict (called after each entry) |
| `_append_batch_results_to_file(new, path)` | Merge new batch results into existing file |
| `_identify_unevaluated_entries(entries, path)` | Return entries whose `entry_id` is not in existing results |

---

## Derived Evaluator Classes

### NEPA Document Analysis

#### `QAEvaluator` (`qa_evaluator.py`)
- **Task**: `question_answer`
- **Entry type**: `QABenchmarkEntry`
- **Supported context types** (`context_type` kwarg):

  | Value | Context source |
  |---|---|
  | `none` | No context — direct LLM query |
  | `gold` | Human-annotated context from benchmark entry |
  | `pdf` | Full PDF reconstructed from JSON chunks |
  | `rag` | ChromaDB retrieval-augmented generation |

- Appends yes/no instruction for `closed` question type
- `evaluate_entry` kwargs: `context_type`, `json_directory`, `chromadb_path`, `collection_name`, `doc_id_csv_path`, `embed_model_name`, `max_attempts`, `retry_delay`
- Result keys: `question`, `answer_expected`, `answer_predicted`, `chunks`, `source_nodes`, `context_type`, `file_name`, `question_type`, `entry_id`

#### `InformationExtractor` (`information_extractor.py`)
- **Task**: `information_extraction`
- **Entry type**: `IEBenchmarkEntry`
- Loads document text from JSON chunks when `entry.document_text` is empty
- Validates responses by `field_type` (`string`, `integer`, `date`, `array_string`)
- `evaluate_entry` kwargs: `json_directory`, `max_attempts`, `retry_delay`
- Result keys: `field_name`, `answer_expected`, `answer_predicted`, `file_name`, `page_number`, `entry_id`

#### `StructuredExtractor` (`structured_extractor.py`)
- **Task**: `structured_extraction`
- **Entry type**: `StructuredIEBenchmarkEntry`
- Generates a `HierarchicalPydanticGenerator` schema from `entry.schema_json` and calls the LLM with `response_format`
- Falls back to extracting JSON from markdown code blocks when needed
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `entry_id`, `answer_expected`, `answer_predicted`

#### `TribeExtractor` (`tribe_extractor.py`)
- **Task**: `tribe_extraction`
- **Entry type**: `TribalBenchmarkEntry`
- Extracts document section by name or number; truncates to token limit before prompting
- Always validates as `array_string` field type
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `file_name`, `source_type`, `source`, `answer_expected`, `answer_predicted`, `entry_id`

---

### Public Comment Analysis

#### `CommentDelineator` (`comment_delineator.py`)
- **Task**: `comment_delineation`
- **Entry type**: `CommentDelineateBenchmarkEntry`
- Loads full comment text from file on demand; validates that extracted quotes exist verbatim in the source text
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `entry_id`, `answer_predicted` (list of quote strings), `answer_expected`

#### `CommentBinAssigner` (`comment_bin_assigner.py`)
- **Task**: `bin_assignment`
- **Entry type**: `CommentBinBenchmarkEntry`
- Loads bin names and descriptions from a JSON guidance file
- Response matching: exact match → subcategory match (text after `:`) → fuzzy match (`cutoff=0.6`)
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `entry_id`, `answer_predicted` (bin name string), `answer_expected`

#### `CommentClassifier` (`comment_classifier.py`)
- **Task**: `comment_classification`
- **Entry type**: `CommentClassificationBenchmarkEntry`
- Binary in-scope / out-of-scope classification; loads project description from file on demand
- Handles affirmative/negative terms and scope-specific phrases
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `entry_id`, `answer_predicted` (bool), `answer_expected`

#### `BinSummarizer` (`bin_summarizer.py`)
- **Task**: `bin_summarization`
- **Entry type**: `BinSummarizerBenchmarkEntry`
- Formats a list of comments into a comments block for the prompt; cleans HTML decorators from response
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`
- Result keys: `entry_id`, `answer_predicted` (cleaned summary text), `answer_expected`

---

### Geographic Analysis

#### `MapClassifier` (`map_classifier.py`)
- **Task**: `map_classification`
- **Entry type**: `MapClassifyBenchmarkEntry`
- Multimodal: `prepare_prompt` returns `(prompt, [image_path])`; image is passed to `llm_handler.generate_response(images=...)`
- Response matching: exact match → subcategory match → fuzzy match (same logic as `CommentBinAssigner`)
- `evaluate_entry` kwargs: `max_attempts`, `retry_delay`, `image_directory`
- Result keys: `entry_id`, `answer_predicted` (class name string), `answer_expected`

---

## Creating a New Evaluator

### 1. Create the evaluator class

```python
# evaluation/new_task_evaluator.py
from evaluation.base_evaluator import BaseEvaluator
from utils import NewTaskBenchmarkEntry, LogManager
from typing import Dict, Any, Optional, List, Tuple

logger = LogManager.get_logger(__name__)

class NewTaskEvaluator(BaseEvaluator):

    def prepare_prompt(self, entry: NewTaskBenchmarkEntry, **kwargs) -> Tuple[str, Optional[List[str]]]:
        # Populate any lazy-loaded entry fields here, e.g.:
        # entry.update_field('computed_field', compute(entry))
        prompt = self.prompt_manager.format_prompt(entry)
        return prompt, None   # return image paths instead of None for multimodal tasks

    def evaluate_entry(self, entry: NewTaskBenchmarkEntry, **kwargs) -> Optional[Dict[str, Any]]:
        max_attempts = kwargs.get('max_attempts', 10)
        retry_delay = kwargs.get('retry_delay', 10)

        prompt, images = self.prepare_prompt(entry, **kwargs)
        if not prompt:
            return None

        def get_response():
            return self.llm_handler.generate_response(prompt, images=images)

        # Add retry / validation logic here
        response = get_response()
        if response is None:
            return None

        return {
            "entry_id": entry.entry_id,
            "answer_expected": entry.expected_answer,
            "answer_predicted": response.strip(),
        }

    def extract_response(self, entry: NewTaskBenchmarkEntry, response: str) -> Optional[Dict[str, Any]]:
        # Called by execute_gemini_batch_job for each raw batch response
        if not response:
            return None
        return {
            "entry_id": entry.entry_id,
            "answer_expected": entry.expected_answer,
            "answer_predicted": response.strip(),
        }
```

### 2. Register in `__init__.py`

```python
from .new_task_evaluator import NewTaskEvaluator

__all__ = [
    # ... existing entries ...
    'NewTaskEvaluator',
]
```

### 3. Create a prompt template

Create `prompts/new_task_prompt.txt` inside `nepabench_directory`:

```
{field_from_entry}

{another_field}

Instructions:
...
```

Placeholder names must match the attribute names on the benchmark entry class.

### 4. Create a config file

```yaml
# configs/new_task_config.yaml
model:
  provider: Vertex
  name: gemini-2.5-pro
  max_tokens: 100000

task: new_task

nepabench_directory: /path/to/benchmark/folder

benchmark:
  input_file: NewTaskBench/benchmark.json
  field_mapping:
    entry_id: entry_id
    field_from_entry: json_key_1
    another_field: json_key_2

evaluation:
  prompt_file: 'prompts/new_task_prompt.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: 10

output:
  directory: 'results/nepabench_tests/new_task_gemini/'
  logfile: 'logs/nepabench_tests/new_task_gemini.log'
```

### 5. Wire into `main.py`

```python
elif conf["task"] == "new_task":
    from evaluation import NewTaskEvaluator
    eval_client = NewTaskEvaluator(llm_client, eval_config['prompt_file'])
```

---

## Common Patterns

### Retry logic

Each evaluator implements a `process_*_with_retries` helper that wraps a `get_response` callable:

```python
def process_response_with_retries(self, get_response_func, validate_func,
                                  max_attempts=10, retry_delay=10):
    for attempt in range(max_attempts):
        try:
            response = get_response_func()
            result = validate_func(response)
            if result is not None:
                return result
            logger.warning(f"Invalid response on attempt {attempt + 1}/{max_attempts}")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_attempts - 1:
            time.sleep(retry_delay)
    return None
```

### Batch processing (sequential)

```python
results_path = evaluator.evaluate_batch(
    entries=benchmark_entries,
    output_path="results/task_results.json",
    continue_from_previous=True,
    max_attempts=10,
    retry_delay=30
)
```

### Batch processing (Gemini batch API)

```python
results_path = evaluator.execute_gemini_batch_job(
    entries=benchmark_entries,
    output_path="results/task_results.json",
    continue_from_previous=True,
    bucket_name="maple_evaluations",
    upload_file_path="input_data/prompts.jsonl",
    predictions_file_path="responses/"
)
```

---

## Dependencies

| Package / Module | Used for |
|---|---|
| `llm_handlers` | LLM provider interface |
| `utils` | Benchmark entry classes, `PromptManager`, `LogManager`, validation utilities |
| `pydantic` | Structured output schema generation (`StructuredExtractor`) |
| `llama_index` | RAG chat client (`QAEvaluator`), LlamaIndex LLM wrappers |
| `difflib` | Fuzzy bin/class name matching (`CommentBinAssigner`, `MapClassifier`) |
| `json`, `os`, `time` | Standard library |
