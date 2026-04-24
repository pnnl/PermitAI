# Utils Directory

This directory contains utility scripts and benchmark entry classes for the MAPLE framework.

## Overview

The `utils/` directory provides core functionality for:
- Loading and managing benchmark entries from JSON/CSV files
- Defining data structures for different NEPA evaluation tasks
- Processing responses from language models
- Managing prompts and templates
- Validating comment quotes and extracted information
- Handling and parsing PDF documents and schema validation

## Benchmark Entry Scripts

### Core Benchmark Entry Classes

Each benchmark entry class is a dataclass that represents a specific NEPA evaluation task:

#### NEPA Fact Retrieval Task
- **Task Name**: `question_answer`
- **Script**: `utils/qa_benchmark_entry.py` 
- **Benchmark Entry Class**: `QABenchmarkEntry`
- **Purpose**: Question-answering tasks on documents, RAG database and selected excerpts/contexts
- **Key Fields**: `question`, `answer`, `question_type`, `file_name`, `context`, `page_number`

#### NEPA Information Extraction Tasks
1. Single information extraction from documents (preferably documents that are AI-ready)
    - **Task Name**: `information_extraction`
    - **Script**: `utils/ie_benchmark_entry.py` 
    - **Benchmark Entry Class**: `IEBenchmarkEntry`
    - **Purpose**: Information extraction tasks for specific fields from NEPA documents
    - **Key Fields**: `field_name`, `field_type`, `description`, `answer`, `document_text`, `chunks_json`, `pages`

2. Structured information extraction from text
    - **Task Name**: `structured_extraction`
    - **Script**: `utils/sie_benchmark_entry.py` 
    - **Benchmark Entry Class**: `StructuredIEBenchmarkEntry`
    - **Purpose**: Structured information extraction from federal notices using predefined schemas
    - **Key Fields**: `text`, `info`, `schema_json`, `instruction_block`

3. Tribe name extraction from documents (preferably documents that are AI-ready)
    - **Task Name**: `tribe_extraction`
    - **Script**: `utils/tribal_benchmark_entry.py` 
    - **Benchmark Entry Class**: `TribalBenchmarkEntry`
    - **Purpose**: Tribe name extraction for tribal consultation and cultural resource analysis
    - **Key Fields**: `file_name`, `source_type`, `source_caption`, `tribes`

#### Public Comment Analysis Entries
1. Comment Delineation
    - **Task Name**: `comment_delineation`
    - **Script**: `utils/comment_delineate_benchmark_entry.py` 
    - **Benchmark Entry Class**: `CommentDelineateBenchmarkEntry`
    - **Purpose**: Comment delineation and identification from NEPA public comments
    - **Key Fields**: `comment_quotes`, `comment_file`, `full_comment`

2. Comment Bin Assignment (text-classification task)
    - **Task Name**: `bin_assignment`
    - **Script**: `utils/bin_assign_benchmark_entry.py` 
    - **Benchmark Entry Class**: `CommentBinBenchmarkEntry`
    - **Purpose**: Assigning public comments to one out of a predefined set of bins based on the concern highlighted in the comment
    - **Key Fields**: `bin`, `comment`, `binning_guidance_file`, `all_bins`, `bins_guidance_block`, `concern`

3. Comment Bin Summarization (text-summarization task)
    - **Task Name**: `bin_summarization`
    - **Script**: `utils/bin_summarizer_benchmark_entry.py` 
    - **Benchmark Entry Class**: `BinSummarizerBenchmarkEntry`
    - **Purpose**: Summarize the comments in bins for NEPA public comment review and analysis 
    - **Key Fields**: `bin`, `comments`, `summary`, `comments_block`

4. Comment Classification (text-classification task)
    - **Task Name**: `comment_classification`
    - **Script**: `utils/comment_classifier_benchmark_entry.py` 
    - **Benchmark Entry Class**: `CommentClassificationBenchmarkEntry`
    - **Purpose**: Classify public comments into in-scope or out-of-scope for a given project with project description. 
    - **Key Fields**: `comment`, `in_scope`, `project_description`, `project_file`
    - `` - 

#### Geographic Map Classification Entries
- **Task Name**: `map_classification`
- **Script**: `utils/map_classify_benchmark_entry.py` 
- **Benchmark Entry Class**: `MapClassifyBenchmarkEntry`
- **Purpose**: Classify an image file from NEPA document into a given set of classes, map-types or map-themes 
- **Key Fields**: `correct_class`, `image_file`, `all_classes`, `project`, `project_summary`, `pdf_name`


### Common BenchmarkEntry Methods

All benchmark entry classes implement these standard methods:

```python
# Convert entry to dictionary
entry_dict = entry.to_dict()

# Create entry from dictionary  
entry = BenchmarkEntryClass.from_dict(data_dict)

# Update field values
entry.update_field('field_name', new_value)
```

## Core Utility Scripts

### `dataloader.py`
**Primary Functions:**
- `load_benchmark_entries(config)` - Main entry point for loading benchmark data
- `load_benchmark_entries_from_json()` - Load entries from JSON files with field mapping
- `load_qa_entries_from_csv()` - Load QA entries from CSV files
- `validate_field_mapping()` - Validate field mappings against data sources

**Key Features:**
- Field mapping between benchmark entry attributes and input file fields
- Question type filtering for QA tasks
- Comprehensive validation and error handling
- Supports CSV format along with JSON for only QA benchmark

### `prompt_utils.py` - PromptManager
**Primary Functions:**
- `format_prompt(entry)` - Replace placeholders with benchmark entry values
- `load_prompt_template(path)` - Load templates from files
- `get_placeholder_info()` - Analyze template placeholders

**Key Features:**
- Template-based prompt generation with `{placeholder}` syntax
- Code block preservation (```code``` sections remain unformatted)
- Metadata field access for complex entry structures

### `response_utils.py`
**Primary Functions:**
- `validate_extracted_information()` - Validate LLM responses against expected formats
- `extract_json_from_response()` - Parse JSON from model responses
- Response cleaning and standardization utilities

### `quote_validator.py`
**Primary Functions:**
- `validate_quotes_in_text()` - Verify quote accuracy in LLM responses
- Quote extraction and validation against source documents

### Document Processing Utils

#### `pdf_utils.py`
**Primary Functions:**
- `get_pdf_context_from_json()` - Extract relevant document chunks for evaluation
- PDF text processing and chunk management

#### `parser_utils.py`
**Primary Functions:**
- `extract_section_by_name()` - Extract document sections by header names
- `extract_section_by_number()` - Extract numbered document sections
- Document structure parsing utilities

#### `rag_utils.py`
**Primary Functions:**
- `setup_rag_chat_client()` - Initialize RAG (Retrieval-Augmented Generation) systems
- Document indexing and retrieval utilities

### Schema and Data Processing

#### `schema_utils.py` - HierarchicalPydanticGenerator
**Primary Functions:**
- `generate_instruction()` - Generate schema-based instructions
- `pydantic_to_vertex_schema()` - Convert Pydantic models to Vertex AI schemas
- Dynamic Pydantic class generation from JSON schemas

#### `image_utils.py`
**Primary Functions:**
- Image processing utilities for multimodal evaluations
- Image format standardization and validation

### Logging and Plotting

#### `logging_utils.py` - LogManager
**Primary Functions:**
- `setup_logging()` - Configure logging for the evaluation framework
- `LogManager.initialize()` - Initialize centralized logging
- `LogManager.get_logger()` - Get configured logger instances

#### `plot_utils.py`
**Primary Functions:**
- Evaluation result visualization utilities
- Performance metric plotting and chart generation

## Creating a New Benchmark Entry

To create a new benchmark entry for a custom NEPA task:

### 1. Create the Entry Class

```python
# new_task_benchmark_entry.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class NewTaskBenchmarkEntry:
    """
    Benchmark entry for [describe your task].
    
    Required Attributes:
        task_specific_field1: Description
        task_specific_field2: Description
        
    Optional Attributes:
        entry_id: Unique identifier
        metadata: Additional metadata
    """
    # Required fields
    task_specific_field1: str
    task_specific_field2: str
    
    # Optional fields
    entry_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Counter for auto-generating IDs
    _id_counter = 0
    
    def __post_init__(self):
        """Generate entry_id if not provided."""
        if not self.entry_id:
            self.__class__._id_counter += 1
            self.entry_id = f"new_task_{self.__class__._id_counter}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            'task_specific_field1': self.task_specific_field1,
            'task_specific_field2': self.task_specific_field2,
            'entry_id': self.entry_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewTaskBenchmarkEntry':
        """Create entry from dictionary."""
        # Extract core fields
        core_fields = {k: v for k, v in data.items() 
                      if k in ['task_specific_field1', 'task_specific_field2', 'entry_id']}
        
        # Extract metadata
        metadata = {k: v for k, v in data.items() 
                   if k not in core_fields and k != 'metadata'}
        
        if 'metadata' in data:
            metadata.update(data['metadata'])
            
        return cls(**core_fields, metadata=metadata)
    
    def update_field(self, attr_name: str, value: Any) -> None:
        """Update field value."""
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        elif attr_name in self.metadata:
            self.metadata[attr_name] = value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{attr_name}'")
```

### 2. Update Imports

Add your new entry class to `utils/__init__.py`:

```python
from .new_task_benchmark_entry import NewTaskBenchmarkEntry

__all__ = [
    # ... existing imports ...
    'NewTaskBenchmarkEntry',
    # ... 
]
```

### 3. Update Dataloader

Add your entry class to the imports and type hints in `dataloader.py`:

```python
from utils.new_task_benchmark_entry import NewTaskBenchmarkEntry

# Update type hints in function signatures to include your new class
```

### 4. Create Configuration

Create a YAML config file in `configs/` with field mapping:

```yaml
benchmark:
  task: 'new_task'
  input_type: 'json'
  input_file: 'input/benchmark/new_task_benchmark.json'
  field_mapping:
    task_specific_field1: 'json_field_1'
    task_specific_field2: 'json_field_2'
    entry_id: 'id'
```

## Usage Examples

### Loading Benchmark Entries
```python
from omegaconf import OmegaConf
from utils import load_benchmark_entries

# Load configuration
config = OmegaConf.load('configs/qa_config.yaml')

# Load benchmark entries
entries = load_benchmark_entries(config)
```

### Using PromptManager
```python
from utils import PromptManager

# Load template and format prompt
prompt_mgr = PromptManager.from_file('input/prompts/qa_template.txt')
formatted_prompt = prompt_mgr.format_prompt(qa_entry)
```

### Validating Responses
```python
from utils import validate_extracted_information

# Validate LLM response
is_valid, errors = validate_extracted_information(
    response_text, expected_format, field_type
)
```

## Dependencies

The utils directory relies on:
- **pydantic**: Data validation and serialization
- **omegaconf**: Configuration management  
- **pandas**: CSV data processing
- **llama_index**: RAG functionality
- **json, re, typing**: Standard library modules

## Integration

The utils directory integrates with:
- `evaluation/` - Evaluators use benchmark entries and utilities
- `configs/` - YAML configurations define field mappings
- `input/benchmark/` - JSON benchmark files loaded by dataloader
- `llm_handlers/` - Response processing utilities