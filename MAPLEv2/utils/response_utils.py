import sys

try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import re, ast
import json
import os
from typing import Dict, Any, Optional, List, Callable, Tuple, Type
from pydantic import BaseModel
import time
from datetime import datetime
from difflib import get_close_matches

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_response_processor.log")

logger = LogManager.get_logger(__name__)

def validate_extracted_information(response: str, field_type: str = "string") -> Tuple[bool, Any]:
    """Validate and format response based on field_type."""
    response = response.strip()
    logger.info(f"Generated response: {response}")
    
    # Handle empty responses
    if not response or "not found" in response.lower():
        return False, None
        
    # Process based on field type
    if field_type == "integer":
        # Extract and validate integer
        try:
            # Find numbers in the text
            num_match = re.search(r'(\d+)', response)
            if num_match:
                value = int(num_match.group(1))
                return True, value
            else:
                logger.warning(f"No integer found in response: {response}")
                return False, None
        except ValueError:
            logger.warning(f"Failed to convert to integer: {response}")
            return False, None
            
    elif field_type == "date":
        # Extract and format date
        try:
            # Try different date formats
            date_formats = [
                # Look for various date patterns
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or similar
                r'(\w+ \d{1,2},? \d{4})',  # Month DD, YYYY
                r'(\d{1,2} \w+ \d{4})'     # DD Month YYYY
            ]
            
            for pattern in date_formats:
                date_match = re.search(pattern, response)
                if date_match:
                    date_str = date_match.group(1)
                    # Try to parse the date
                    try:
                        # Try to handle various formats
                        if '/' in date_str or '-' in date_str:
                            # Try MM/DD/YYYY or similar formats
                            if len(date_str.split('/')[-1]) == 2 or len(date_str.split('-')[-1]) == 2:
                                # Handle two-digit year (MM/DD/YY)
                                date_obj = datetime.strptime(date_str.replace('-', '/'), '%m/%d/%y')
                            else:
                                # Handle four-digit year (MM/DD/YYYY)
                                date_obj = datetime.strptime(date_str.replace('-', '/'), '%m/%d/%Y')
                        elif ',' in date_str:
                            # Try "Month DD, YYYY" format
                            date_obj = datetime.strptime(date_str, '%B %d, %Y')
                        elif '.' in date_str:
                            # Try "MM.DD.YY" format
                            date_obj = datetime.strptime(date_str, '%m.%d.%Y')
                        else:
                            # Try other common formats
                            for fmt in ['%m-%d-%Y', '%B %d %Y', '%d %B %Y', '%Y-%m-%d', '%m/%d/%y']:
                                try:
                                    date_obj = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                                    
                        # Format to MM-DD-YYYY
                        formatted_date = date_obj.strftime('%m-%d-%Y')
                        return True, formatted_date
                    except (ValueError, UnboundLocalError):
                        continue
            
            logger.warning(f"No valid date found in: {response}")
            return False, None
        except Exception as e:
            logger.warning(f"Error processing date: {str(e)}")
            return False, None
            
    elif field_type == "array_string":
        # Handle list of strings
        try:
            # Pattern to match the list inside code blocks
            pattern = r'```(?:python|json)?\s*(\[.*?\])\s*```'
            
            match = re.search(pattern, response, re.DOTALL)
            if match:
                list_string = match.group(1)
                # Use ast.literal_eval to safely parse the Python list
                try:
                    parsed_list = ast.literal_eval(list_string)
                    return True, parsed_list
                except (ValueError, SyntaxError):
                    return False, None
            
            # Check if response is already in list format
            bracket_match = re.search(r'\[(.*?)\]', response)
            if bracket_match:
                try:
                    # Extract the content inside brackets and try to parse it
                    bracket_content = '[' + bracket_match.group(1) + ']'
                    import json
                    value = json.loads(bracket_content)
                    if isinstance(value, list) and all(isinstance(item, str) for item in value):
                        return True, list(set(value))
                except json.JSONDecodeError:
                    # If JSON parsing fails, try simple splitting
                    content = bracket_match.group(1)
                    items = list(set([item.strip().strip('"\'') for item in content.split(',') if item.strip()]))
                    if items:
                        return True, items
            
            # Split by common separators
            for sep in [',', ';', '|', '\n']:
                if sep in response:
                    items = list(set([item.strip() for item in response.split(sep) if item.strip()]))
                    if items:
                        return True, items
            
            # If no separators found, treat as single item list
            return True, [response]
            
        except Exception as e:
            logger.warning(f"Error processing array: {str(e)}")
            return False, None
    
    else:  # Default: string
        # For string type, just return as is
        return True, response

def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from an LLM response.
    
    Args:
        response (str): The LLM response text.
        
    Returns:
        Optional[Dict[str, Any]]: The extracted JSON object or None if extraction failed.
    """
    # Look for JSON content in the response
    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
    
    if not json_match:
        logger.warning("No JSON content found in response")
        return None
    
    json_str = json_match.group(1)
    
    try:
        json_obj = json.loads(json_str)
        logger.info(f"Successfully extracted JSON with {len(json_obj)} fields")
        return json_obj
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON content: {str(e)}")
        return None
