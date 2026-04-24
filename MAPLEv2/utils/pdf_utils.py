import os
import sys
import json
import tiktoken
from typing import List


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_pdf_utils.log")

logger = LogManager.get_logger(__name__)

def count_tokens(text) -> int:
    """
    Count the number of tokens in a given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def truncate_text(text, max_tokens) -> str:
    """
    Truncate the input text to a specified maximum number of tokens.

    Args:
        text (str): The input text to truncate.
        max_tokens (int): The maximum number of tokens to keep.

    Returns:
        str: The truncated text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def get_pdf_context_from_json(json_path: str, pages: List[str]=None, max_tokens: int = 100000) -> str:
    """
    Get the entire PDF content from the extracted information in the JSON files.

    Args:
        json_path (str): Directory containing the JSON files with PDF chunks.
        pages (List[str], optional): List of page keys to load PDF context
        max_tokens (int, optional): Maximum number of tokens to include in the context.

    Returns:
        str: The PDF context.

    Raises:
        ValueError: If required parameters are missing.
        FileNotFoundError: If the JSON file for the PDF is not found.
    """
    if not json_path:
        logger.error("JSON directory path must be provided when using PDF context.")
        raise ValueError("Need to provide a path to the JSON directory with PDF chunks.")
    
    if not os.path.exists(json_path):
        logger.error(f"JSON file {json_path} does not exist")
        raise FileNotFoundError(f"JSON file {json_path} with PDF chunks not found")

    # Extract text from the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

        pdf_context = ""
        if pages:
            logger.info("Loading PDF content from provided pages")
            for page_key in pages:
                try:
                    pdf_context += data[page_key]['text']
                except Exception as e:
                    logger.error(f"Error adding content for {page_key}: {e}")
        else:
            for page_key in data:
                try:
                    pdf_context += data[page_key]['text']
                except Exception as e:
                    logger.error(f"Error adding content for {page_key}: {e}")
    
    # Truncate the input text if it exceeds the max_tokens limit
    token_count = count_tokens(pdf_context)
    logger.info(f"Current token count = {token_count}")
    if token_count > max_tokens:
        logger.info(f"Truncating the context because it exceeds maximum token count {max_tokens}")
        pdf_context = truncate_text(pdf_context, max_tokens)
    return pdf_context