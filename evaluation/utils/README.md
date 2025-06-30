# The `utils` directory

This directory contains utility functions used across the project.

## Contents
1. [Data Loading and Processing](#data-loading-and-processing)
2. [Logging Utilities](#logging-utilities)
3. [Plotting Utilities](#plotting-utilities)

## Data Loading and Processing

The `utils/dataloader.py` script provides robust functionality for loading and processing benchmark and evaluation data. It includes flexible data structures and methods for handling various CSV formats.

### Data Structures

1. **BenchmarkEntry**
   - Represents a single benchmark question entry
   - Required fields:
     - `question`: The question text
     - `answer`: The expected answer
   - Optional fields:
     - `file_name`: Source document name (required for 'pdf' and 'rag' context types)
     - `page_number`: Page number in the source document
     - `question_type`: Type of question (needed for type-specific analysis)
     - `proof`: Supporting evidence or explanation
     - `context`: Context information (required for 'gold' context type)

2. **ResponseEntry**
   - Represents a collection of evaluation responses
   - Required fields:
     - `questions`: List of questions
     - `answers`: List of predicted answers
     - `ground_truths`: List of correct answers
     - `contexts`: List of context information
   - Optional fields:
     - `files`: List of source document names
     - `pages`: List of page numbers
     - `qtypes`: List of question types



### Benchmark Data Processing

The `process_benchmark_csv` function provides comprehensive benchmark data loading with automatic context type detection.

```python
from utils.dataloader import process_benchmark_csv

# Load benchmark data and get supported context types
entries, context_types = process_benchmark_csv("benchmark.csv")
print(f"Supported context types: {context_types}")
```

#### Context Type Detection

The function automatically detects supported context types based on available columns:

| Context Type | Required Column | Description |
|-------------|----------------|-------------|
| `none`      | -             | Always supported, requires no special columns |
| `rag`       | `file_name`   | For retrieval-augmented generation |
| `pdf`       | `file_name`   | For full PDF context |
| `gold`      | `context`     | For provided ground truth context |

#### Column Requirements

1. Required Columns:
   - `question`: The question text
   - `answer`: The expected answer

2. Optional Columns with Features:
   - `file_name`: Enables RAG and PDF context types
   - `context`: Enables Gold context type
   - `question_type`: Enables question type filtering and analysis
   - `page_number`: Adds page information to results
   - `proof`: Adds proof/explanation to entries

#### Question Type Filtering

```python
# Single question type
entries, types = process_benchmark_csv("benchmark.csv", question_type="closed")

# Multiple question types
entries, types = process_benchmark_csv("benchmark.csv", 
                                     question_type=["closed", "open"])
```

#### Usage Example

```python
# Load benchmark data
entries, context_types = process_benchmark_csv("benchmark.csv")

# Check available context types
if 'rag' in context_types:
    print("RAG evaluation is supported")
if 'gold' in context_types:
    print("Gold context evaluation is supported")

# Use with evaluator
evaluator = Evaluator(...)
if 'rag' in context_types:
    evaluator.evaluate_benchmark(
        entries,
        context_type='rag',
        ...
    )
```

#### Error Handling

The script includes comprehensive error handling:
- Raises `ValueError` for missing required columns
- Issues warnings for missing optional columns
- Provides detailed logging of processed data
- Handles edge cases like missing file extensions

1. Missing Required Columns:
```python
try:
    entries, types = process_benchmark_csv("benchmark.csv")
except ValueError as e:
    print(f"Error: {e}")  # e.g., "Missing required columns: question, answer"
```

2. Context Type Validation:
```python
entries, types = process_benchmark_csv("benchmark.csv")
if desired_context_type not in types:
    print(f"Warning: {desired_context_type} context type not supported")
```

#### Best Practices

1. Context Type Handling:
   ```python
   entries, types = process_benchmark_csv("benchmark.csv")
   
   # Check before evaluation
   if desired_context_type not in types:
       raise ValueError(f"{desired_context_type} not supported with current data")
   ```

2. Column Verification:
   ```python
   # Verify specific features are available
   if 'rag' in types and 'gold' in types:
       print("Both RAG and Gold context evaluations are possible")
   ```

3. Error Recovery:
   ```python
   try:
       entries, types = process_benchmark_csv("benchmark.csv")
   except ValueError:
       # Fall back to minimal evaluation
       print("Using basic evaluation without context")
       context_type = 'none'
   ```

This enhanced functionality helps ensure that evaluations are only attempted with supported context types based on the available data columns.

### Response Data Processing

The `load_results_csv` method loads evaluation results from a CSV file for RAGAS evaluation.

```python
results_data = load_results_csv("evaluation_results.csv")
```

Features:
- Handles required and optional columns flexibly
- Aggregates context chunks automatically
- Provides clear warnings for missing optional data

### Dependencies

- pandas
- numpy
- typing (Python standard library)
- warnings (Python standard library)
- logging (via LogManager)



## Logging Utilities

The `utils/logging_utils.py` script provides a centralized logging configuration system through the `LogManager` class. This utility ensures consistent logging across the entire project with proper formatting and log level management.

### LogManager Features

#### Initialization

The `LogManager` class provides a singleton pattern for logging configuration:

```python
from utils.logging_utils import LogManager

# Initialize logging for your module
LogManager.initialize("logs/my_module.log")

# Get a logger for your module
logger = LogManager.get_logger(__name__)
```

#### Log Format

Default log format includes:
- Timestamp
- Log level
- Module name
- Message
- Additional context (when provided)

Example log output:
```
2024-03-15 10:30:45,123 - INFO - mymodule - Starting process with parameters: x=10, y=20
2024-03-15 10:30:45,456 - WARNING - mymodule - Missing optional parameter 'z'
2024-03-15 10:30:45,789 - ERROR - mymodule - Process failed: Division by zero
```

### Usage Examples

1. Basic module logging:
```python
from utils.logging_utils import LogManager

# Initialize logging at the start of your script
LogManager.initialize("logs/module.log")

# Get a logger for your module
logger = LogManager.get_logger(__name__)

# Use the logger
logger.info("Process started")
logger.warning("Resource usage high")
logger.error("An error occurred", exc_info=True)
```

2. Logging with context:
```python
# Log with additional context
parameters = {"batch_size": 32, "learning_rate": 0.001}
logger.info(f"Training started with parameters: {parameters}")

# Log errors with traceback
try:
    # Your code here
    pass
except Exception as e:
    logger.error("Processing failed", exc_info=True)
```

3. Using in multiple modules:
```python
# In main.py
LogManager.initialize("logs/main.log")

# In other modules, just get the logger
logger = LogManager.get_logger(__name__)
```

### Configuration Options

The `LogManager` supports the configuration options provided in the `configs/logging_config.ini` file.

### Log Levels

Supported log levels in order of severity:
1. `DEBUG` - Detailed information for debugging
2. `INFO` - General information about program execution
3. `WARNING` - Indicate a potential problem
4. `ERROR` - A more serious problem
5. `CRITICAL` - A critical problem that may prevent program execution

```python
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")
```

### Integration with Project Components

The logging utility is integrated with various project components:

1. **Evaluator Integration**:
```python
from utils.logging_utils import LogManager

logger = LogManager.get_logger(__name__)

class Evaluator:
    def evaluate_benchmark(self, ...):
        logger.info("Starting benchmark evaluation")
        # ... evaluation code ...
        logger.info("Evaluation completed")
```

2. **Data Loading Integration**:
```python
logger.info(f"Loading benchmark data from {csv_path}")
logger.warning("Missing optional column 'context'")
logger.error("Failed to load CSV file", exc_info=True)
```

### Best Practices

1. Initialize logging once at the application entry point:
```python
if __name__ == "__main__":
    LogManager.initialize("logs/app.log")
```

2. Use appropriate log levels:
- `DEBUG` for detailed troubleshooting
- `INFO` for general operational events
- `WARNING` for unexpected but handleable situations
- `ERROR` for serious problems
- `CRITICAL` for fatal errors

3. Include relevant context in log messages:
```python
logger.info(f"Processing batch {batch_num}/{total_batches}")
```

4. Use error logging with tracebacks:
```python
try:
    # Your code
    pass
except Exception as e:
    logger.error(f"Failed to process data: {str(e)}", exc_info=True)
```

### Directory Structure

The logger automatically creates the necessary directory structure:
```
project_root/
├── logs/
│   ├── main.log
│   ├── evaluation.log
│   └── processing.log
└── utils/
    └── logging_utils.py
```

## Plotting Utilities

The `utils/plot_utils.py` script provides comprehensive visualization tools for analyzing evaluation results through heatmaps and bar plots. It includes two main classes: `HeatmapGenerator` and `BarPlotGenerator`.

### HeatmapGenerator

The `HeatmapGenerator` class offers methods to create heatmap visualizations comparing metrics across different dimensions.

#### Multi-Context Heatmap Generation

```python
from utils.plot_utils import HeatmapGenerator

HeatmapGenerator.generate_matrix_heatmap_multiple_context(
    path_pfx="results/model",
    model="gpt4",
    model_name="GPT-4",
    context=["none", "rag", "gold"],
    metric="answer_correctness",
    output_path="heatmap_contexts.png",
    percentage=True  # Display values as percentages
)
```

Customization options:
```python
# Additional kwargs for customization
kwargs = {
    "figsize": (12, 6),            # Figure size
    "labelsize": 18,               # Label font size
    "ticklabelsize": 15,           # Tick label size
    "xtickrotation": 45,           # X-axis label rotation
    "annotation_size": 15,         # Annotation text size
    "colormap": "Blues",           # Colormap for heatmap
    "percentage": False            # Display as percentages
}
```

#### Single Context, Multiple Models Heatmap

```python
HeatmapGenerator.generate_matrix_heatmap_single_context(
    path_pfx="results",
    models=["gpt4", "mistral", "gemini"],
    metric="answer_similarity",
    output_path="heatmap_models.png",
    context_type="gold",
    xticklabels=["GPT-4", "Mistral", "Gemini"]
)
```

### BarPlotGenerator

The `BarPlotGenerator` class creates bar plots to compare metrics across different models.

```python
from utils.plot_utils import BarPlotGenerator

BarPlotGenerator.generate_bar_plot(
    path_pfx="results",
    models=["gpt4", "mistral", "gemini"],
    metric="answer_correctness",
    context_type="gold",
    output_path="barplot_scores.png",
    xticklabels=["GPT-4", "Mistral", "Gemini"]
)
```

### Common Features

#### Metric Processing
- Automatic handling of closed-question correctness
- Support for percentage-based visualization
- Consistent color schemes and formatting

#### Customization Options
```python
plot_kwargs = {
    # Figure properties
    "figsize": (12, 6),           # Width, height in inches
    "labelsize": 18,              # Size of axis labels
    "ticklabelsize": 15,          # Size of tick labels
    
    # Label customization
    "xticklabels": ["Model A", "Model B"],  # Custom x-axis labels
    "xtickrotation": 45,          # Rotation angle for x labels
    "annotation_size": 15,        # Size of value annotations
    
    # Display options
    "colormap": "Blues",          # Color scheme for heatmaps
    "percentage": True,           # Show values as percentages
    
    # Axis labels
    "xlabel": "Custom X Label",
    "ylabel": "Custom Y Label"
}
```

### Usage Examples

1. Comparing contexts for a single model:
```python
HeatmapGenerator.generate_matrix_heatmap_multiple_context(
    path_pfx="results/evaluation",
    model="gpt4",
    model_name="GPT-4 Turbo",
    context=["none", "rag", "gold"],
    metric="answer_correctness",
    output_path="context_comparison.png",
    percentage=True,
    xtickrotation=45
)
```

2. Comparing models for a specific context:
```python
HeatmapGenerator.generate_matrix_heatmap_single_context(
    path_pfx="results/evaluation",
    models=["gpt4", "mistral", "gemini"],
    metric="answer_similarity",
    output_path="model_comparison.png",
    context_type="gold",
    xticklabels=["GPT-4", "Mistral-7B", "Gemini"],
    percentage=True
)
```

3. Creating bar plots for model comparison:
```python
BarPlotGenerator.generate_bar_plot(
    path_pfx="results/evaluation",
    models=["gpt4", "mistral", "gemini"],
    metric="answer_correctness",
    context_type="gold",
    output_path="model_scores.png",
    xticklabels=["GPT-4", "Mistral", "Gemini"],
    percentage=True,
    xtickrotation=45
)
```

### File Organization

Expected file structure for evaluation results:
```
results/
├── model_gpt4/
│   ├── scores_none.csv
│   ├── scores_rag.csv
│   └── scores_gold.csv
├── model_mistral/
│   └── ...
└── model_gemini/
    └── ...
```

### Dependencies
- matplotlib
- seaborn
- pandas
- numpy

### Best Practices

1. Consistent naming:
```python
# Use consistent prefixes and suffixes
path_pfx = "results/eval"
context_type = "gold"
output_path = "visualizations/heatmap.png"
```

2. Error handling:
- Checks for missing metrics
- Validates file existence
- Provides informative logging

3. Customization:
- Use kwargs for fine-tuning visualizations
- Maintain consistent styling across plots
- Provide clear labels and annotations

4. Output management:
- Creates output directories if needed
- Automatically adds .png extension if missing
- Closes figures to manage memory