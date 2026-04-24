import sys

try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import os
import json
import pandas as pd
from typing import Type, Dict, List, Optional, Union, Any
import warnings
import inspect
from omegaconf import DictConfig, ListConfig

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_dataloader.log")

logger = LogManager.get_logger(__name__)

# Import the dataclasses
from utils.qa_benchmark_entry            import QABenchmarkEntry
from utils.ie_benchmark_entry            import IEBenchmarkEntry
from utils.tribal_benchmark_entry        import TribalBenchmarkEntry
from utils.sie_benchmark_entry           import StructuredIEBenchmarkEntry
from utils.bin_assign_benchmark_entry    import CommentBinBenchmarkEntry
from utils.map_classify_benchmark_entry  import MapClassifyBenchmarkEntry
from utils.comment_classifier_benchmark_entry import CommentClassificationBenchmarkEntry
from utils.bin_summarizer_benchmark_entry import BinSummarizerBenchmarkEntry
from utils.comment_delineate_benchmark_entry import CommentDelineateBenchmarkEntry

def validate_field_mapping(
        field_mapping: Dict[str, str], 
        data_fields: List[str],
        entry_class: Union[
            QABenchmarkEntry, IEBenchmarkEntry, 
            TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            CommentBinBenchmarkEntry, MapClassifyBenchmarkEntry,
            CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry,
            CommentDelineateBenchmarkEntry
            ]
    ) -> List[str]:
    """
    Validate a field mapping against available data fields and entry class attributes.
    
    Args:
        field_mapping (Dict[str, str]): Mapping from entry fields to data fields.
        data_fields (List[str]): Available fields in the data source.
        entry_class: The class to check entry fields against.
    
    Returns:
        List[str]: List of missing data fields.
    
    Warnings/Exceptions:
        Logs warnings for invalid entry fields and missing data fields.
    """
    if not field_mapping:
        return []
        
    # Get all fields from the entry class
    import inspect
    entry_fields = inspect.signature(entry_class).parameters
    
    # Check if all keys in field_mapping are valid entry class attributes
    invalid_keys = [key for key in field_mapping.keys() if key not in entry_fields]
    if invalid_keys:
        warning_msg = f"Invalid keys in field_mapping (not attributes of {entry_class.__name__}): {', '.join(invalid_keys)}"
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)
    
    # Check if field_mapping values exist in data fields
    missing_fields = [data_field for data_field in field_mapping.values() 
                     if data_field not in data_fields]
    if missing_fields:
        warning_msg = f"Field mapping points to non-existent data fields: {', '.join(missing_fields)}"
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)
        
    return missing_fields

def load_benchmark_entries_from_json(
        input_benchmark_file, 
        entry_class: Type, entry_class_name: str, 
        field_mapping: Optional[Dict[str, str]] = None,
        question_types: Optional[List[str]] = None,
        nepabench_directory: Optional[str] = None
        ) -> List[Union[QABenchmarkEntry, IEBenchmarkEntry, StructuredIEBenchmarkEntry,
                        BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                        CommentDelineateBenchmarkEntry,MapClassifyBenchmarkEntry, TribalBenchmarkEntry]]:
    """
    Load benchmark entries from JSON.

    Parameters:
        input_benchmark_file (str): Path to the benchmark JSON file containing the benchmark entries.
        entry_class (Type): Class name of the Benchmark Entry
        entry_class_name (str): name of the task for the benchmark entry
        field_mapping (Optional[Dict[str, str]]): Mapping between BenchmarkEntry dataclass fields and benchmark JSON attributes.
            Key: field name in BenchmarkEntry dataclass
            Value: corresponding attribute name in benchmark JSON
            If None, field names will be used as attribute names.
        nepabench_directory: Directory path to NEPABench. Default is None
    
    Returns:
        List[Type]: List of parsed benchmark entries.
    """
    if not os.path.exists(input_benchmark_file):
        raise FileNotFoundError(f"Input JSON file not found: {input_benchmark_file}")
    
    # Get all fields from the Benchmark Entry dataclass
    benchmark_entry_fields = inspect.signature(entry_class).parameters
    required_fields = [name for name, param in benchmark_entry_fields.items() 
                      if param.default == inspect.Parameter.empty]
    
    # Create default field mapping if not provided
    if field_mapping is None:
        field_mapping = {field: field for field in benchmark_entry_fields.keys() if field in required_fields}
    
    logger.info(f"Loading {entry_class_name} benchmark entries from: {input_benchmark_file}")
    logger.info(f"Field mapping: {field_mapping}")
    
    # Check if all required fields have mappings
    missing_mappings = [field for field in required_fields if field not in field_mapping]
    if missing_mappings:
        raise ValueError(f"Missing field mappings for required fields: {', '.join(missing_mappings)}")
    
    # Check if the benchmark entry is capable of filtering question type
    if question_types and not hasattr(entry_class, 'question_type'):
        logger.warning(f"The benchmark entry class {entry_class.__name__} for {entry_class_name} task does not have question_type attribute")
        question_types = None
    
    # Load entries from JSON file
    entries = []
    try:
        with open(input_benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Get all unique attribute names from all JSON entries
        all_json_attrs = set()
        for item in data:
            all_json_attrs.update(item.keys())
            
        # Validate field mapping against JSON attributes and Benchmark Entry fields
        validate_field_mapping(field_mapping, list(all_json_attrs), entry_class)
        
        # Process each entry in the JSON
        for item in data:
            
            # Check for required fields in JSON based on mapping
            item_has_required_fields = True
            for field in required_fields:
                json_attr = field_mapping[field]
                if json_attr not in item:
                    logger.warning(f"Skipping entry missing required field '{json_attr}': {item}")
                    item_has_required_fields = False
                    break
                    
            if not item_has_required_fields:
                continue
            
            # Map JSON attributes to Benchmark Entry fields
            entry_data = {}
            for entry_field, json_attr in field_mapping.items():
                if json_attr in item:
                    entry_data[entry_field] = item[json_attr]
            
            # Create Benchmark Entry object
            try:
                entry = entry_class(**entry_data)
                if question_types:
                    if entry.question_type not in question_types:
                        continue
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Error creating entry: {str(e)} - {item}")
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {input_benchmark_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading entries: {e}")
        raise
    
    if nepabench_directory:
        entries = update_entries_nepabench(entries, entry_class_name, nepabench_directory)
        logger.info(f"Added NEPABench directory path as prefix to necessary fields in {len(entries)}.")

    logger.info(f"Loaded {len(entries)} {entry_class_name} benchmark entries")
    return entries


def load_qa_entries_from_csv(
    csv_path: str, 
    field_mapping: Optional[Dict[str, str]] = None,
    question_types: Optional[Union[str, List[str]]] = None
) -> List[QABenchmarkEntry]:
    """
    Load QA benchmark entries from a CSV file with optional filtering by question type.
    
    Args:
        csv_path (str): Path to the CSV file.
        field_mapping (Optional[Dict[str, str]]): Mapping between QABenchmarkEntry fields and CSV columns.
            Key: field name in QABenchmarkEntry
            Value: corresponding column name in CSV
            If None, field names will be used as column names.
        question_types (Optional[Union[str, List[str]]]): Question type(s) to filter by.
            If None, all question types will be included.
    
    Returns:
        List[QABenchmarkEntry]: List of QA benchmark entries.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If required columns are missing.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Get required fields from the QABenchmarkEntry class
    import inspect
    qa_fields = inspect.signature(QABenchmarkEntry).parameters
    required_fields = [name for name, param in qa_fields.items() 
                      if param.default == inspect.Parameter.empty]
    
    # Create default field mapping if not provided
    if field_mapping is None:
        field_mapping = {field: field for field in qa_fields.keys()}
    
    logger.info(f"Loading QA benchmark entries from CSV: {csv_path}")
    logger.info(f"Field mapping: {field_mapping}")
    
    # Check if all required fields have mappings
    missing_mappings = [field for field in required_fields if field not in field_mapping]
    if missing_mappings:
        raise ValueError(f"Missing field mappings for required fields: {', '.join(missing_mappings)}")
    
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Validate field mapping against CSV columns and QABenchmarkEntry fields
    validate_field_mapping(field_mapping, list(df.columns), QABenchmarkEntry)
    
    # Check required columns based on mapping
    required_columns = {field_mapping[field] for field in required_fields}
    missing_required_columns = required_columns - set(df.columns)
    if missing_required_columns:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_required_columns)}")
    
    # Filter by question_type(s) if specified
    question_type_column = field_mapping.get('question_type', 'question_type')
    if question_types is not None and question_type_column in df.columns:
        if isinstance(question_types, str):
            # Single question type
            df = df[df[question_type_column] == question_types]
            logger.info(f"Filtering for question type: {question_types}")
        else:
            # List of question types
            df = df[df[question_type_column].isin(question_types)]
            logger.info(f"Filtering for question types: {', '.join(question_types)}")
            
        # Log the number of questions found for each type
        if not df.empty:
            type_counts = df[question_type_column].value_counts()
            for qtype, count in type_counts.items():
                logger.info(f"Found {count} questions of type '{qtype}'")
        else:
            warning = f"No questions found for specified question type(s): {question_types}"
            logger.warning(warning)
            warnings.warn(warning, UserWarning)
            
    elif question_types is not None and question_type_column not in df.columns:
        warning = f"Cannot filter by question_type as the column '{question_type_column}' is not present in the CSV."
        logger.warning(warning)
        warnings.warn(warning, UserWarning)
    
    # Convert DataFrame rows to QABenchmarkEntry objects
    entries = []
    for _, row in df.iterrows():
        # Map CSV columns to QABenchmarkEntry fields
        entry_data = {}
        for qa_field, csv_column in field_mapping.items():
            if csv_column in df.columns:
                entry_data[qa_field] = row[csv_column]
                
        # Convert 'metadata' to dict if it exists and is a string
        if 'metadata' in entry_data and isinstance(entry_data['metadata'], str):
            try:
                entry_data['metadata'] = json.loads(entry_data['metadata'])
            except json.JSONDecodeError:
                entry_data['metadata'] = {}
                
        # Create QABenchmarkEntry
        try:
            entry = QABenchmarkEntry(**entry_data)
            entries.append(entry)
        except Exception as e:
            logger.warning(f"Error creating entry from row {_}: {str(e)} - {row}")
    
    logger.info(f"Loaded {len(entries)} QA benchmark entries from CSV")
    return entries

def update_entries_nepabench(
        entries: List[Union[
            QABenchmarkEntry, IEBenchmarkEntry, 
            TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry,
            MapClassifyBenchmarkEntry,CommentDelineateBenchmarkEntry]], 
        task: str,
        nepabench_directory: str = None
        ):
    if not nepabench_directory:
        return entries
    
    else:
        updated_entries = []
        for entry in entries:
            if task == 'information_extraction' or task == 'question_answer':
                entry.update_field('chunks_json', os.path.join(nepabench_directory, entry.chunks_json))
            elif task == 'tribe_extraction':
                entry.update_field('file_name', os.path.join(nepabench_directory, entry.file_name))
            elif task == 'structured_extraction':
                entry.update_field('schema_json', os.path.join(nepabench_directory, entry.schema_json))
            elif task == 'bin_assignment':
                entry.update_field('binning_guidance_file', os.path.join(nepabench_directory, entry.binning_guidance_file))
            elif task == 'comment_delineation':
                entry.update_field('comment_file', os.path.join(nepabench_directory, entry.comment_file))

            # Create the updated entry list
            updated_entries.append(entry)
        return updated_entries

def load_benchmark_entries(config: Union[DictConfig, ListConfig]) -> List[Union[
    QABenchmarkEntry, IEBenchmarkEntry, 
    TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
    CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry,
    MapClassifyBenchmarkEntry,CommentDelineateBenchmarkEntry
    ]]:
    """
    Load benchmark entries based on configuration settings.
    
    Args:
        config (Union[DictConfig, ListConfig]): Configuration object containing benchmark settings.
            Expected structure:
            benchmark:
              task: 'question_answer' or 'information_extraction' or 'tribe_extraction' or 'structured_extraction' or 'bin_assignment' or 'comment_classification' or 'bin_summarization' or 'map_classification' or 'comment_delineation'
              input_type: 'json' or 'csv'
              input_file: 'path/to/input_file'
              field_mapping: (optional for both tasks)
                field1: json_attr1
                field2: json_attr2
              question_types: (optional for 'question_answer' task)
                - type1
                - type2
    
    Returns:
        List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry, CommentDelineateBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry, BinSummarizerBenchmarkEntry, MapClassifyBenchmarkEntry]]: List of benchmark entries.
        
    Raises:
        ValueError: If configuration is missing required fields or contains invalid values.
        FileNotFoundError: If input file doesn't exist.
    """
    # Validate config structure
    if not isinstance(config, (DictConfig, ListConfig)):
        raise ValueError("Config must be a DictConfig or ListConfig object")
    
    # Extract benchmark settings
    try:
        nepabench_directory = config.get('nepabench_directory', None)
        benchmark_config = config.benchmark
        task = config.get('task', 'question_answer').lower()  # Default to 'question_answer'
        input_type = benchmark_config.input_file.split('.')[-1].lower()  # Default to 'json'
        input_file = os.path.join(nepabench_directory, benchmark_config.input_file)
    except AttributeError as e:
        raise ValueError(f"Missing required configuration field: {str(e)}")
    
    logger.info(f"Loading benchmark entries. Task: {task}, Input type: {input_type}, File: {input_file}")
    
    # Extract field mapping if provided
    field_mapping = None
    if 'field_mapping' in benchmark_config:
        # Convert OmegaConf mapping to Python dict
        field_mapping = dict(benchmark_config.field_mapping)
        logger.info(f"Using field mapping from config: {field_mapping}")
    
    # Load entries based on task and input type
    if task == 'question_answer':
        # Get question_types if provided
        question_types = None
        if 'question_types' in benchmark_config:
            question_types = list(benchmark_config.question_types)
            
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, QABenchmarkEntry, task, field_mapping, question_types, nepabench_directory=nepabench_directory)
        
        elif input_type == 'csv':
            # Delegate to the CSV loader with field mapping
            return load_qa_entries_from_csv(input_file, field_mapping, question_types)
        
        else:
            raise ValueError(f"Invalid input_type for QA task: {input_type}. Supported types: 'json', 'csv'")
    
    elif task == 'information_extraction':
        if input_type == 'json':
            # Delegate to the IE JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, IEBenchmarkEntry, task, field_mapping, nepabench_directory=nepabench_directory)
        else:
            logger.error(f"Invalid input_type for information_extraction task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for information_extraction task: {input_type}. Only 'json' is supported.")
    
    elif task == 'tribe_extraction':
        if input_type == 'json':
            # Delegate to the Tribal JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, TribalBenchmarkEntry, task, field_mapping, nepabench_directory=nepabench_directory)
        else:
            logger.error(f"Invalid input_type for tribe_extraction task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for tribe_extraction task: {input_type}. Only 'json' is supported.")
    
    elif task == 'structured_extraction':
        if input_type == 'json':
            # Delegate to the Fed JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, StructuredIEBenchmarkEntry, task, field_mapping, nepabench_directory=nepabench_directory)
        else:
            logger.error(f"Invalid input_type for structured_extraction task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for structured_extraction task: {input_type}. Only 'json' is supported.")
    
    elif task == 'comment_delineation':
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, CommentDelineateBenchmarkEntry, task, field_mapping, nepabench_directory=nepabench_directory)
        else:
            logger.error(f"Invalid input_type for comment_delineation task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for comment_delineation task: {input_type}. Only 'json' is supported.")
    
    elif task == 'bin_assignment':
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, CommentBinBenchmarkEntry, task, field_mapping, nepabench_directory=nepabench_directory)
        else:
            logger.error(f"Invalid input_type for bin_assignment task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for bin_assignment task: {input_type}. Only 'json' is supported.")
    
    elif task == 'comment_classification':
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, CommentClassificationBenchmarkEntry, task, field_mapping)
        else:
            logger.error(f"Invalid input_type for comment_classification task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for comment_classification task: {input_type}. Only 'json' is supported.")
        
    elif task == 'bin_summarization':
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, BinSummarizerBenchmarkEntry, task, field_mapping)
        else:
            logger.error(f"Invalid input_type for bin_summarization task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for bin_summarization task: {input_type}. Only 'json' is supported.")
        
    elif task == 'map_classification':
        if input_type == 'json':
            # Delegate to the JSON loader with field mapping
            return load_benchmark_entries_from_json(input_file, MapClassifyBenchmarkEntry, task, field_mapping)
        else:
            logger.error(f"Invalid input_type for map_classification task: {input_type}. Only 'json' is supported.")
            raise ValueError(f"Invalid input_type for map_classification task: {input_type}. Only 'json' is supported.")

    else:
        raise ValueError(f"Invalid task: {task}. Supported tasks: 'question_answer', 'information_extraction', 'tribe_extraction', 'structured_extraction', 'bin_assignment', 'map_classification', 'comment_classification', 'bin_summarization', 'comment_delineation'")


# For testing the module
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    args = parser.parse_args()
    
    from omegaconf import OmegaConf
    conf = OmegaConf.load(args.config)
    entries = load_benchmark_entries(conf)
    print(entries[:2])
