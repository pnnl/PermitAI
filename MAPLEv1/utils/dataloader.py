import os
import sys
import pandas as pd
import numpy as np
from typing import List, NamedTuple, Optional, Union, Set, Tuple
import warnings

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_dataloader.log")

logger = LogManager.get_logger(__name__)

class BenchmarkEntry(NamedTuple):
    """
    A named tuple representing a single entry in the benchmark dataset.

    Required Attributes:
        question (str): The actual question text.
        answer (str): The correct answer to the question.

    Optional Attributes:
        file_name (Optional[str]): The name of the file containing the entry.
        page_number (Optional[int]): The page number where the entry is located.
        question_type (Optional[str]): The type of question.
        proof (Optional[str]): The proof or explanation for the answer.
        context (Optional[str]): The context or background information for the question.
    """
    question: str
    answer: str
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    question_type: Optional[str] = None
    proof: Optional[str] = None
    context: Optional[str] = None

class ResponseEntry(NamedTuple):
    """
    A named tuple representing a collection of responses for evaluation.

    Required Attributes:
        questions (List[str]): List of questions.
        answers (List[str]): List of predicted answers.
        ground_truths (List[str]): List of correct answers.
        contexts (List[List[str]]): List of contexts for each question.

    Optional Attributes:
        files (Optional[List[str]]): List of file names.
        pages (Optional[List[int]]): List of page numbers.
        qtypes (Optional[List[str]]): List of question types.
    """
    questions: List[str]
    answers: List[str]
    ground_truths: List[str]
    contexts: List[List[str]]
    files: Optional[List[str]] = None
    pages: Optional[List[int]] = None
    qtypes: Optional[List[str]] = None


def extract_unique_ids(csv_path: str) -> dict:
    """
    Extract unique_id for each pdf_name from the given CSV file.
    Remove double quotes from unique_id values.
    
    Args:
        csv_path (str): Path to the CSV file containing the mapping.
    
    Returns:
        Dict[str, str]: A dictionary mapping pdf_name to unique_id.
        Returns an empty dictionary if the file is empty or doesn't contain required columns.
    """
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return {}
    
    # Check if the DataFrame is empty or missing required columns
    if df.empty or not all(col in df.columns for col in ['pdf_name', 'unique_id']):
        return {}
    
    # Remove double quotes from unique_id
    df['unique_id'] = df['unique_id'].str.replace('"', '')
    return dict(zip(df['pdf_name'], df['unique_id']))

def process_benchmark_csv(file_path: str, question_type: Optional[Union[str, List[str]]] = None) -> Tuple[List[BenchmarkEntry], Set[str]]:
    """
    Load a CSV file, filter by question type(s), and return benchmark entries.
    Also identifies possible context types based on available columns.

    Args:
        file_path (str): Path to the CSV file.
        question_type (Optional[Union[str, List[str]]], optional): A single question type or list of question types to filter by.

    Returns:
        Tuple[List[BenchmarkEntry], Set[str]]: A tuple containing:
            - List of BenchmarkEntry objects
            - Set of supported context types ('none', 'rag', 'pdf', 'gold') based on available columns

    Example:
        >>> entries, context_types = process_benchmark_csv("questions.csv", question_type="closed")
        >>> print(f"Supported context types: {context_types}")
        Supported context types: {'none', 'gold'}
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Check required columns
    required_columns = {'question', 'answer'}
    optional_columns = {
        'file_name': "Cannot run evaluation pipeline for 'pdf' and 'rag' context types without 'file_name' column.",
        'context': "Cannot run evaluation pipeline for 'gold' context type without 'context' column.",
        'question_type': "Cannot generate summarization plots for different question types without 'question_type' column.",
        'page_number': "Page number information will not be available in the results.",
        'proof': "Proof information will not be available in the results."
    }

    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

    # Determine supported context types based on available columns
    supported_context_types = {'none'}  # 'none' is always supported
    
    if 'file_name' in df.columns:
        supported_context_types.add('rag')
        supported_context_types.add('pdf')
        logger.info("'file_name' column found - RAG and PDF context types are supported")
    else:
        logger.warning("'file_name' column not found - RAG and PDF context types are not supported")
    
    if 'context' in df.columns:
        supported_context_types.add('gold')
        logger.info("'context' column found - Gold context type is supported")
    else:
        logger.warning("'context' column not found - Gold context type is not supported")

    # Check other optional columns and issue warnings
    for col, warning_msg in optional_columns.items():
        if col not in df.columns:
            warning = f"Column '{col}' not found in CSV. {warning_msg}"
            logger.warning(warning)
            warnings.warn(warning, UserWarning)

    # Filter by question_type(s) if specified and the column exists
    if question_type is not None and 'question_type' in df.columns:
        if isinstance(question_type, str):
            df = df[df['question_type'] == question_type]
            logger.info(f"Filtering for question type: {question_type}")
        else:
            df = df[df['question_type'].isin(question_type)]
            logger.info(f"Filtering for question types: {', '.join(question_type)}")
            
        # Log the number of questions found for each type
        if not df.empty:
            type_counts = df['question_type'].value_counts()
            for qtype, count in type_counts.items():
                logger.info(f"Found {count} questions of type '{qtype}'")
        else:
            warning = f"No questions found for specified question type(s): {question_type}"
            logger.warning(warning)
            warnings.warn(warning, UserWarning)
            
    elif question_type is not None:
        warning = "Cannot filter by question_type as the column is not present in the CSV."
        logger.warning(warning)
        warnings.warn(warning, UserWarning)

    # Process entries
    results = []
    for _, row in df.iterrows():
        entry_data = {
            'question': row['question'],
            'answer': row['answer']
        }

        # Add optional fields if they exist
        if 'file_name' in df.columns:
            file_name = row['file_name']
            if isinstance(file_name, str) and not file_name.endswith('.pdf'):
                file_name = f"{file_name}.pdf"
            entry_data['file_name'] = file_name

        if 'page_number' in df.columns:
            entry_data['page_number'] = row['page_number']
        if 'question_type' in df.columns:
            entry_data['question_type'] = row['question_type']
        if 'proof' in df.columns:
            entry_data['proof'] = row['proof']
        if 'context' in df.columns:
            entry_data['context'] = row['context']

        results.append(BenchmarkEntry(**entry_data))

    logger.info(f"Processed {len(results)} benchmark entries")
    logger.info(f"Supported context types: {', '.join(sorted(supported_context_types))}")
    
    return results, supported_context_types

def aggregate_chunks(x):
    """
    Aggregate three chunks of text into a numpy array.

    Args:
        x (pd.Series): A pandas Series containing 'chunk_1', 'chunk_2', and 'chunk_3'.

    Returns:
        np.ndarray: A numpy array containing the three chunks as strings.
    """
    return np.array([str(x['chunk_1']), str(x['chunk_2']), str(x['chunk_3'])])

def load_results_csv(file_path: str) -> ResponseEntry:
    """
    Load a results CSV file and return a RAGAS-ready dataset.

    Args:
        file_path (str): Path to the results CSV file.

    Returns:
        ResponseEntry: A ResponseEntry object containing the loaded data.

    Raises:
        ValueError: If required columns are missing.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Define required and optional columns
    required_columns = {
        'question', 'answer_predicted', 'answer_expected', 
        'chunk_1', 'chunk_2', 'chunk_3'
    }
    
    optional_columns = {
        'file_name': "File name information will not be included in the response entry.",
        'page_number': "Page number information will not be included in the response entry.",
        'question_type': "Question type information will not be included in the response entry."
    }

    # Check for required columns
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in the CSV file: {', '.join(missing_required)}")

    # Check optional columns and issue warnings
    for col, warning_msg in optional_columns.items():
        if col not in df.columns:
            warning = f"Column '{col}' not found in CSV. {warning_msg}"
            logger.warning(warning)
            warnings.warn(warning, UserWarning)
    
    # Create empty lists for optional data
    files = df['file_name'].tolist() if 'file_name' in df.columns else None
    pages = df['page_number'].tolist() if 'page_number' in df.columns else None
    qtypes = df['question_type'].tolist() if 'question_type' in df.columns else None
    
    # Create the ResponseEntry
    dataset = ResponseEntry(
        files=files,
        pages=pages,
        qtypes=qtypes,
        questions=df['question'].tolist(),
        answers=[str(x) for x in df['answer_predicted'].tolist()],
        ground_truths=df['answer_expected'].tolist(),
        contexts=df.apply(lambda row: aggregate_chunks(row), axis=1).tolist()
    )

    return dataset
