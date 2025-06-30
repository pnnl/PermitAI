# Evaluation Components

This section describes the evaluation modules used for benchmarking and assessing LLM responses.

## Response Generation Evaluator (`evaluator.py`)

The `Evaluator` class provides functionality for generating and evaluating LLM responses using different context types.

### Features

- Multiple context types support:
  - `none`: Direct query-response without context
  - `pdf`: Uses entire PDF content as context
  - `rag`: Retrieval-Augmented Generation with document filtering
  - `gold`: Uses provided gold-standard context

- Flexible response generation with retry mechanism
- Support for batch processing
- Progress persistence

### Usage

```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(
    llm_handler=llm_handler,
    prompt_file="input/prompts/with_context.txt"
)

# Generate single response
response, chunks, sources = evaluator.generate_response(
    benchmark_entry,
    context_type="rag",
    chromadb_path="input/database",
    collection_name="eis",
    docID_csvpath="input/metadata.csv",
    embed_model="BAAI/bge-small-en-v1.5",
)

# Process benchmark dataset
evaluator.evaluate_benchmark(
    benchmark_data=benchmark_entries,
    output_path="results/responses_rag.csv",
    context_type="rag",
    chromadb_path="input/database",
    collection_name="eis",
    docID_csvpath="input/metadata.csv",
    embed_model="BAAI/bge-small-en-v1.5",
    continue_from_previous=True
)
```

### Context Type Requirements

1. `pdf` context:
   - Requires `file_name` in benchmark entries
   - Needs JSON directory with PDF chunks

2. `rag` context:
   - Requires `file_name` in benchmark entries
   - Needs ChromaDB path and collection name
   - Uses document filtering based on file names

3. `gold` context:
   - Requires `context` in benchmark entries
   - Uses provided context directly

### Output Format

Results CSV columns (in order):
```
file_name,page_number,question_type,question,
answer_expected,answer_predicted,
chunk_1,chunk_2,chunk_3,
source_1,source_2,source_3
```

## RAGAS Metrics Evaluator (`metric.py`)

The `RAGAS_Evaluator` class provides comprehensive evaluation of LLM responses using RAGAS metrics.

### Available Metrics

- `answer_similarity`: Measures similarity between predicted and expected answers
- `answer_correctness`: Evaluates correctness of predictions
- `context_precision`: Measures precision of retrieved contexts
- `context_recall`: Evaluates recall of relevant information
- `faithfulness`: Assesses response faithfulness to context

### Usage

```python
from evaluation.metric import RAGAS_Evaluator

# Initialize with specific metrics
evaluator = RAGAS_Evaluator(
    results_csv_path="results/responses.csv",
    llm_handler=llm_handler,
    embedding_model_name="BAAI/bge-small-en-v1.5",
    metrics=['answer_similarity', 'answer_correctness']
)

# Evaluate results
evaluator.evaluate_results(
    output_csv_path="results/scores.csv",
    batch_size=20
)
```

### Features

- Selective metric evaluation
- Batch processing support
- Progress persistence
- Ordered output format

### Output Format

Results CSV columns (in order):
```
file_name,page_number,question_type,question,
answer_expected,answer_predicted,
[selected_metrics...]
```

### Configuration

```python
# Initialize with custom metrics
evaluator = RAGAS_Evaluator(
    results_csv_path="results/scores.csv",
    llm_handler=llm_handler,
    embedding_model_name="BAAI/bge-small-en-v1.5",
    metrics=[
        'answer_similarity',
        'answer_correctness',
    ]
)

# Use all available metrics
evaluator = RAGAS_Evaluator(
    results_csv_path="results/scores.csv",
    llm_handler=llm_handler,
    embedding_model_name="BAAI/bge-small-en-v1.5"
)
```

## Integration Example

Complete evaluation pipeline:

```python
# 1. Generate responses
response_evaluator = Evaluator(
    llm_handler=llm_handler,  # load the llm handler prior to this call
    docID_csvpath="input/metadata.csv",
    embed_model="BAAI/bge-small-en-v1.5",
    prompt_file="input/prompts/with_context.txt"
)

response_evaluator.evaluate_benchmark(
    benchmark_data=benchmark_entries,  # load the benchmark entries prior to this call
    output_path="results/responses_rag.csv",
    context_type="rag",
    chromadb_path="input/database",
    collection_name="eis",
    continue_from_previous=True
)

# 3. Evaluate responses using RAGAS
metrics_evaluator = RAGAS_Evaluator(
    results_csv_path="results/responses_rag.csv",
    llm_handler=eval_llm_handler,
    embedding_model_name="BAAI/bge-small-en-v1.5",
    metrics=['answer_similarity', 'context_precision']
)

metrics_evaluator.evaluate_results(
    output_csv_path="results/scores_rag.csv",
    batch_size=20
)
```

## Best Practices

1. Response Generation:
   - Use appropriate context type based on your use case
   - Ensure required attributes are present in benchmark entries
   - Consider using `continue_from_previous=True` for long evaluations

2. Metrics Evaluation:
   - Select metrics relevant to your evaluation needs
   - Use appropriate batch sizes based on available resources
   - Monitor evaluation progress through logs

3. Error Handling:
   - Handle missing attributes appropriately
   - Monitor logs for evaluation progress and errors
   - Use try-except blocks for robust error handling

## Dependencies

- RAGAS metrics library
- Hugging Face Embeddings
- ChromaDB (for RAG context)
- pandas, numpy
- logging utilities

## Additional Resources

- [RAGAS Metrics Documentation](#https://docs.ragas.io/en/stable/concepts/metrics/overview/)
- [Embedding Models](#https://huggingface.co/)