# Configuration Files

This directory contains YAML configuration files for the MAPLE evaluation framework. These configuration files control the behavior of the framework for different evaluation tasks and models.

## Overview

The configuration files follow a structured format with several main sections:

- `model`: Specifies the LLM provider and model name
- `nepabench_directory`: Absolute path to the NEPABench dataset root
- `task`: Defines the evaluation task type
- `benchmark`: Contains settings for loading benchmark data
- `evaluation`: Determines how benchmark entries are processed
- `scoring`: Configures metrics for evaluation (task-dependent)
- `output`: Specifies where results and logs are saved

## Available Configuration Files

| File | Task | Benchmark |
|---|---|---|
| `cx_gemini.yaml` | `information_extraction` | CXBench (Categorical Exclusion) |
| `ea_gemini.yaml` | `information_extraction` | EABench (Environmental Assessment) |
| `eis_gemini.yaml` | `information_extraction` | EISBench (Environmental Impact Statement) |
| `tribe_gemini.yaml` | `tribe_extraction` | TribeBench |
| `delineate_gemini.yaml` | `comment_delineation` | CommentDelineate |
| `assign_gemini.yaml` | `bin_assignment` | BinAssign |
| `summarize_gemini.yaml` | `bin_summarization` | BinSummarize |
| `fedreg_gemini.yaml` | `structured_extraction` | FedRegBench |
| `qa_gold_gemini.yaml` | `question_answer` | nepaquad (gold context) |
| `qa_none_gemini.yaml` | `question_answer` | nepaquad (no context) |
| `qa_pdf_gemini.yaml` | `question_answer` | nepaquad (PDF context) |
| `qa_rag_gemini.yaml` | `question_answer` | nepaquad (RAG context) |

## Task Types

The framework currently supports the following task types:

1. **`information_extraction`** — Extract structured field values from NEPA documents (CX, EA, EIS)
2. **`tribe_extraction`** — Identify tribal nations mentioned in NEPA documents
3. **`comment_delineation`** — Delineate individual comments within public comment text
4. **`bin_assignment`** — Assign public comments to predefined bins using guidance
5. **`bin_summarization`** — Summarize grouped comments within a bin
6. **`structured_extraction`** — Extract structured data from Federal Register notices using a schema
7. **`question_answer`** — Answer questions about NEPA documents (supports multiple context types)

## Configuration Structure

### Common Settings

All configuration files include these sections:

```yaml
model:
  provider: Vertex             # LLM provider (Vertex, Azure, AWS, HuggingFace)
  name: gemini-2.5-pro         # Model identifier
  max_tokens: 100000           # Maximum context length

nepabench_directory: /rcfs/projects/policyai/nepabench  # Dataset root

output:
  directory: <path>            # Directory to store results
  logfile: <path>              # Path for log file
```

All `benchmark.input_file`, `evaluation.prompt_file`, and `output` paths are resolved relative to `nepabench_directory`.

### Information Extraction Task

Used by `cx_gemini.yaml`, `ea_gemini.yaml`, and `eis_gemini.yaml`.

```yaml
task: information_extraction

benchmark:
  input_file: <Bench>/benchmark.json
  field_mapping:
    file_name: file_name
    field_type: type
    chunks_json: json_file
    entry_id: entry_id
    field_name: field
    description: description
    answer: answer
    page_number: page_number   # present in EA/EIS, absent in CX

evaluation:
  prompt_file: 'prompts/<task>_prompt.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: <int>         # seconds between retries

scoring:
  metrics_config: "metrics/<task>_metrics.json"
```

### Tribe Extraction Task

Used by `tribe_gemini.yaml`.

```yaml
task: tribe_extraction

benchmark:
  input_file: TribeBench/benchmark.json
  field_mapping:
    file_name: file_name
    source_type: source_type
    source_caption: source_caption
    tribes: tribes

evaluation:
  prompt_file: 'prompts/tribe_extract.txt'
  continue_previous: false
  eval_kwargs:
    max_attempts: 10

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

### Comment Delineation Task

Used by `delineate_gemini.yaml`.

```yaml
task: comment_delineation

benchmark:
  input_file: CommentDelineate/benchmark.json
  field_mapping:
    comment_quotes: quotes
    comment_file: comment_text

evaluation:
  prompt_file: 'prompts/comment_delineate.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: 10

scoring:
  metrics:
    soft_precision: ...
    soft_recall: ...
    soft_f1: ...
```

### Bin Assignment Task

Used by `assign_gemini.yaml`.

```yaml
task: bin_assignment

benchmark:
  input_file: BinAssign/benchmark.json
  field_mapping:
    bin: bin
    comment: comment
    binning_guidance_file: guidance

evaluation:
  prompt_file: 'prompts/bin_assign.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: 60

scoring:
  metrics:
    - exact_match
```

### Bin Summarization Task

Used by `summarize_gemini.yaml`.

```yaml
task: bin_summarization

benchmark:
  input_file: BinSummarize/benchmark.json
  field_mapping:
    bin: bin
    comments: comments
    summary: summary

evaluation:
  prompt_file: 'prompts/bin_summary.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: 20

scoring:
  batch_size: 10
  embed_model_name: 'all-MiniLM-L6-v2'
  metrics:
    - answer_correctness
    - answer_similarity
```

### Structured Extraction Task

Used by `fedreg_gemini.yaml`.

```yaml
task: structured_extraction

benchmark:
  input_file: FedRegBench/benchmark.json
  field_mapping:
    text: text
    info: info
    schema_json: schema_json

evaluation:
  prompt_file: 'prompts/sie_prompt.txt'
  continue_previous: true
  eval_kwargs:
    max_attempts: 10
    retry_delay: 30
```

No `scoring` section — evaluation is handled externally.

### Question Answering Task

Used by `qa_gold_gemini.yaml`, `qa_none_gemini.yaml`, `qa_pdf_gemini.yaml`, and `qa_rag_gemini.yaml`.

```yaml
task: question_answer

benchmark:
  input_file: 'nepaquad/benchmark.json'
  field_mapping:
    question: question
    answer: answer
    file_name: file_name
    chunks_json: json_file
    context: context
    question_type: question_type
  question_types:
    - closed
    - evaluation

evaluation:
  prompt_file: 'prompts/<with_context|no_context>.txt'
  continue_previous: true
  eval_kwargs:
    context_type: <none|gold|pdf|rag>
    max_attempts: 10
    # RAG-only additional fields:
    doc_id_csv_path: 'nepaquad/dataset/metadata.csv'
    chromadb_path: nepaquad/dataset/database/
    embed_model_name: 'all-MiniLM-L6-v2'

scoring:
  batch_size: 4
  embed_model_name: 'all-MiniLM-L6-v2'
  metrics:
    - answer_correctness
    - answer_similarity
```

## Context Types for Question Answering

| `context_type` | Description | Prompt file |
|---|---|---|
| `none` | Direct query with no document context | `prompts/no_context.txt` |
| `gold` | Human-annotated context from benchmark entries | `prompts/with_context.txt` |
| `pdf` | Full PDF content provided as context | `prompts/with_context.txt` |
| `rag` | Retrieval-Augmented Generation via ChromaDB | `prompts/with_context.txt` |

## Provider Settings

Different model providers require specific environment variables:

- **Vertex**: Requires GCP service account credentials (`GOOGLE_APPLICATION_CREDENTIALS`)
- **Azure**: Requires `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, etc.
- **AWS**: Requires AWS credentials, region, and role configuration
- **HuggingFace**: Requires local model path

## Usage

To run an evaluation using one of these configuration files, use the `main.py` script:

```bash
python main.py --configPath configs/your_config.yaml
```

You can also specify an environment file with model credentials:

```bash
python main.py --configPath configs/your_config.yaml --env .env
```

## Logging Configuration File

This directory contains `logging_config.ini` to configure Python logging. It defines console and file handlers with timestamp-formatted output for both the root logger and the application logger.
