# Metrics Directory

This directory contains scoring and evaluation modules for the MAPLE framework. Three evaluator classes handle different task types, backed by shared deterministic and fuzzy metric primitives.

## Table of Contents

- [Overview](#overview)
- [Evaluator Classes](#evaluator-classes)
  - [RAGAS_Evaluator](#ragas_evaluator-ragas_metricspy)
  - [MetricsEvaluator](#metricsevaluator-metrics_evaluatorpy)
  - [NestedStructureEvaluator](#nestedstructureevaluator-nested_evaluatorpy)
- [Metric Primitives](#metric-primitives)
  - [Deterministic Metrics](#deterministic-metrics-closed_metricspy)
  - [Fuzzy / Semantic Metrics](#fuzzy--semantic-metrics-fuzzy_metricspy)
- [Task-to-Evaluator Mapping](#task-to-evaluator-mapping)
- [Configuration Reference](#configuration-reference)
- [Output Format](#output-format)
- [Extending the Package](#extending-the-package)
- [Dependencies](#dependencies)

---

## Overview

| Class | Script | Used for |
|---|---|---|
| `RAGAS_Evaluator` | `ragas_metrics.py` | QA and summarization tasks |
| `MetricsEvaluator` | `metrics_evaluator.py` | Information / tribe / comment extraction tasks |
| `NestedStructureEvaluator` | `nested_evaluator.py` | Structured JSON extraction tasks |

---

## Evaluator Classes

### `RAGAS_Evaluator` (`ragas_metrics.py`)

Evaluates QA and summarization responses using the RAGAS framework.

**Constructor**:
```python
RAGAS_Evaluator(llm_handler: BaseLLMHandler, embedding_model_name: str)
```

**Available metrics** (passed as strings to `evaluate_responses`):

| Metric | Description |
|---|---|
| `answer_correctness` | Combined factual + semantic correctness |
| `answer_similarity` | Semantic similarity to reference answer |
| `context_precision` | Fraction of context chunks that are relevant |
| `context_recall` | Fraction of answer supported by context |
| `faithfulness` | Whether answer is grounded in retrieved context |

**Key method**:
```python
evaluator.evaluate_responses(
    response_json_path="results/responses.json",
    output_scores_path="results/scores.json",
    metrics=["answer_correctness", "answer_similarity"],
    batch_size=4,                   # number of entries per RAGAS call
    continue_from_previous=True     # resume interrupted evaluation
)
```

**Usage**:
```python
from metrics import RAGAS_Evaluator
from llm_handlers import AzureChatOpenAIHandler

chat_handler = AzureChatOpenAIHandler("gpt-4-turbo-preview")
ragas_evaluator = RAGAS_Evaluator(
    llm_handler=chat_handler.llm,
    embedding_model_name="all-MiniLM-L6-v2"
)

ragas_evaluator.evaluate_responses(
    response_json_path="results/nepabench_tests/qa_gold_gemini/responses.json",
    output_scores_path="results/nepabench_tests/qa_gold_gemini/scores.json",
    metrics=["answer_correctness", "answer_similarity"],
    batch_size=4,
    continue_from_previous=True
)
```

Scores are saved incrementally after each batch to prevent data loss on interruption. Retries failed batches up to 5 times with exponential backoff (via `tenacity`).

---

### `MetricsEvaluator` (`metrics_evaluator.py`)

General-purpose evaluator that applies deterministic and fuzzy metrics to `(answer_expected, answer_predicted)` pairs. Supports both field-specific metric configs (via JSON file) and global metrics applied to all entries.

**Constructor**: `MetricsEvaluator()` — no arguments.

**Available metrics**:

| Category | Metric name |
|---|---|
| Closed-set | `precision`, `recall`, `f1`, `exact_match` |
| Edit distance | `char_edit_distance`, `word_edit_distance` |
| Numerical | `numerical_error`, `time_difference`, `geo_distance` |
| Semantic similarity | `semantic_similarity_embedding`, `semantic_similarity_fuzzy`, `abbreviation_similarity` |
| Soft set matching | `soft_precision`, `soft_recall`, `soft_f1` |

Soft metrics (`soft_precision`, `soft_recall`, `soft_f1`) use MD5-keyed caching to avoid redundant embedding computations.

**Key method**:
```python
MetricsEvaluator.batch_evaluate(
    response_path: str,
    config: dict,
    output_path: str = None,
    field_name_key: str = "field_name",   # response key holding the field name
    continue_from_previous: bool = True,
    nepabench_directory: str = None        # resolves relative metrics_config paths
)
```

**Config option 1 — field-specific metrics (JSON file)**:

Use when different fields need different metrics (e.g., information extraction).

```yaml
# in YAML config
scoring:
  metrics_config: "metrics/cx_metrics.json"
```

```python
metrics_evaluator.batch_evaluate(
    response_path="results/nepabench_tests/cx_gemini/responses.json",
    config={"metrics_config": "metrics/cx_metrics.json"},
    output_path="results/nepabench_tests/cx_gemini/scores.json",
    field_name_key="field_name",
    nepabench_directory="/path/to/benchmark/folder"
)
```

The JSON file maps field names to metric dictionaries:
```json
{
    "Document_Title": {
        "metrics": {
            "word_edit_distance": {},
            "char_edit_distance": {},
            "semantic_similarity_embedding": {"embedding_model": "all-MiniLM-L6-v2"},
            "semantic_similarity_fuzzy": {}
        }
    },
    "Categorical_Exclusion_Applied": {
        "metrics": {
            "precision": {},
            "recall": {},
            "f1": {}
        }
    },
    "Year": {
        "metrics": {
            "char_edit_distance": {},
            "numerical_error": {}
        }
    }
}
```

Optional `"weights"` key per field applies a weighted average to produce an `aggregate` score:
```json
{
    "Document_Title": {
        "metrics": {
            "semantic_similarity_embedding": {"embedding_model": "all-MiniLM-L6-v2"},
            "word_edit_distance": {}
        },
        "weights": {
            "semantic_similarity_embedding": 0.7,
            "word_edit_distance": 0.3
        }
    }
}
```

**Config option 2 — global metrics (inline)**:

Use when the same metrics apply to all entries (e.g., tribe extraction, comment delineation).

```yaml
# in YAML config
scoring:
  metrics:
    soft_precision:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
    soft_recall:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
    soft_f1:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
```

```python
config = {
    "metrics": {
        "soft_precision": {"embedding_model": "all-MiniLM-L6-v2", "use_abbreviation_matching": True},
        "soft_recall":    {"embedding_model": "all-MiniLM-L6-v2", "use_abbreviation_matching": True},
        "soft_f1":        {"embedding_model": "all-MiniLM-L6-v2", "use_abbreviation_matching": True},
    }
}
metrics_evaluator.batch_evaluate(
    response_path="results/nepabench_tests/tribe_gemini/responses.json",
    config=config,
    output_path="results/nepabench_tests/tribe_gemini/scores.json"
)
```

A metrics list (instead of dict) is also accepted and treated as metrics with no kwargs:
```python
config = {"metrics": ["exact_match"]}   # for bin_assignment
```

---

### `NestedStructureEvaluator` (`nested_evaluator.py`)

Compares nested JSON structures field-by-field. Designed for `structured_extraction` tasks where `answer_expected` and `answer_predicted` are JSON objects.

**Constructor**: `NestedStructureEvaluator(metrics_evaluator: MetricsEvaluator = None)`

Requires no YAML `scoring` configuration — invoked directly from `main.py` without metric kwargs.

**Key methods**:

```python
# Single entry
result = evaluator.evaluate_nested_structure(
    response_entry,
    list_handling='merged',      # MAPLE default
    auto_detect_metrics=True
)

# Batch
results = evaluator.batch_evaluate_nested_structures(
    response_path="results/nepabench_tests/fedreg_gemini/responses.json",
    list_handling='merged',
    output_path="results/nepabench_tests/fedreg_gemini/scores.json"
)
```

**List handling modes**:

| Mode | Behaviour | When to use |
|---|---|---|
| `merged` | All list items' values combined under a single key | Default for MAPLE — treats list fields as unordered sets |
| `indexed` | Each item gets its own key: `field[0].sub`, `field[1].sub` | When order matters |
| `union_keys` | All possible keys across list items, `None`-padded for missing | Partial matching across variable-length lists |

**Auto metric detection** (when `auto_detect_metrics=True`):

| Field name pattern | Detected metric |
|---|---|
| `id`, `uuid` | `exact_match` |
| `date`, `time` | `time_difference` |
| `phone`, `zip` | `numerical_error` |
| `address`, `street`, `city`, `state` | `semantic_similarity` |
| `name`, `person`, `department` | `semantic_similarity` |
| `type` | `exact_match` |
| list values | `soft_f1` |

**Output structure per entry**:
```json
{
    "field_results": {
        "field.path": {
            "expected": "...",
            "predicted": "...",
            "metric_type": "semantic_similarity",
            "score": 0.91
        }
    },
    "overall_statistics": {
        "total_fields": 12,
        "evaluated_fields": 11,
        "failed_evaluations": 1,
        "average_score": 0.87,
        "metric_type_distribution": {"semantic_similarity": 7, "exact_match": 4},
        "list_handling_mode": "merged",
        "list_alignment_enabled": true
    },
    "metric_type_averages": {
        "semantic_similarity": 0.89,
        "exact_match": 0.82
    }
}
```

---

## Metric Primitives

### Deterministic Metrics (`closed_metrics.py`)

| Function | Input types | Returns |
|---|---|---|
| `evaluate_precision(gt, pred)` | list / comma-sep string | float |
| `evaluate_recall(gt, pred)` | list / comma-sep string | float |
| `evaluate_f1(gt, pred)` | list / comma-sep string | float |
| `evaluate_exact_match(gt, pred)` | any | `1.0` or `0.0` |
| `evaluate_char_edit_distance(gt, pred)` | string | float (0–1, lower = worse) |
| `evaluate_word_edit_distance(gt, pred)` | string | float (0–1, lower = worse) |
| `evaluate_numerical_error(gt, pred)` | numeric string / array | float (0–1, lower = worse) |
| `evaluate_time_difference(gt, pred)` | date/time string | float (0–1, lower = worse) |
| `evaluate_geo_distance(gt, pred)` | `"lat,lon"` string | float (normalised distance) |
| `evaluate_llm_match(gt, pred, llm_client)` | string | float |

All functions normalise inputs via `_preprocess_values` (handles strings, lists, comma-separated values) and return a float in `[0, 1]` unless noted otherwise.

### Fuzzy / Semantic Metrics (`fuzzy_metrics.py`)

| Function | Description |
|---|---|
| `semantic_similarity(a, b)` | Embedding cosine similarity (SentenceTransformer) |
| `semantic_similarity_embedding(a, b, embedding_model)` | Same, explicit model |
| `semantic_similarity_fuzzy(a, b)` | RapidFuzz token sort ratio |
| `abbreviation_similarity(a, b)` | Acronym / abbreviation-aware matching |
| `soft_precision_recall(expected, possible, embedding_model, ...)` | Returns `(precision, recall, f1)` tuple for set-valued fields |

`soft_precision_recall` falls back to RapidFuzz if SentenceTransformer is unavailable.

---

## Task-to-Evaluator Mapping

| Task | Evaluator | `scoring` config field |
|---|---|---|
| `question_answer` | `RAGAS_Evaluator` | `metrics` (list), `batch_size`, `embed_model_name` |
| `bin_summarization` | `RAGAS_Evaluator` | `metrics` (list), `batch_size`, `embed_model_name` |
| `structured_extraction` | `NestedStructureEvaluator` | *(none required)* |
| `information_extraction` | `MetricsEvaluator` | `metrics_config` (JSON path) |
| `tribe_extraction` | `MetricsEvaluator` | `metrics` (dict with kwargs) |
| `comment_delineation` | `MetricsEvaluator` | `metrics` (dict with kwargs) |
| `bin_assignment` | `MetricsEvaluator` | `metrics` (list) |
| `comment_classification` | `MetricsEvaluator` | `metrics` (list or dict) |
| `map_classification` | `MetricsEvaluator` | `metrics` (list or dict) |

---

## Configuration Reference

### RAGAS tasks

```yaml
scoring:
  batch_size: 4
  embed_model_name: 'all-MiniLM-L6-v2'
  metrics:
    - answer_correctness
    - answer_similarity
```

### Information extraction (field-specific JSON)

```yaml
scoring:
  metrics_config: "metrics/cx_metrics.json"   # relative to nepabench_directory
```

### Tribe / comment extraction (global soft metrics)

```yaml
scoring:
  metrics:
    soft_precision:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
    soft_recall:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
    soft_f1:
      embedding_model: "all-MiniLM-L6-v2"
      use_abbreviation_matching: true
```

### Bin assignment (exact match)

```yaml
scoring:
  metrics:
    - exact_match
```

### Structured extraction

```yaml
# No scoring section needed — NestedStructureEvaluator requires no configuration.
```

---

## Output Format

**`RAGAS_Evaluator`** and **`MetricsEvaluator`** write a list of scored entries:

```json
[
  {
    "entry_id": "entry_001",
    "answer_expected": "...",
    "answer_predicted": "...",
    "scores": {
      "answer_correctness": 0.82,
      "answer_similarity": 0.91
    }
  }
]
```

When `weights` are configured, a top-level `"aggregate"` key is added alongside `"scores"`.

**`NestedStructureEvaluator`** writes per-field results (see structure above under [NestedStructureEvaluator](#nestedstructureevaluator-nested_evaluatorpy)).

---

## Extending the Package

### Adding a new metric to `MetricsEvaluator`

1. Implement the function in `closed_metrics.py` or `fuzzy_metrics.py`:

```python
def my_metric(expected: Any, predicted: Any, **kwargs) -> float:
    ...
    return score   # float in [0, 1]
```

2. Register it in `MetricsEvaluator.__init__()`:

```python
self.available_metrics["my_metric"] = my_metric
```

3. Reference it in a JSON config or YAML `scoring.metrics`.

### Adding a new task

Wire the appropriate evaluator in `main.py`:

```python
elif conf['task'] == 'my_new_task':
    from metrics import MetricsEvaluator
    metrics_evaluator = MetricsEvaluator()
    metrics_evaluator.batch_evaluate(
        response_path=response_json_path,
        config=OmegaConf.to_container(conf.scoring),
        output_path=scores_json_path,
        nepabench_directory=conf.nepabench_directory
    )
```

---

## Dependencies

| Package | Used for |
|---|---|
| `ragas` | RAGAS metric computation |
| `datasets` | RAGAS dataset preparation |
| `langchain-huggingface` | HuggingFace embeddings for RAGAS |
| `sentence-transformers` | Embedding-based similarity in fuzzy metrics |
| `rapidfuzz` | Fuzzy string matching fallback |
| `tenacity` | Retry with exponential backoff in RAGAS evaluator |
| `numpy` | Weighted averages, NaN handling |
