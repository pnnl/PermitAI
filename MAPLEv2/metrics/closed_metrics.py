import os
import sys
from typing import Dict, List, Any, Union, Set, Tuple, Optional
import nltk
from nltk.metrics import edit_distance
from nltk.tokenize import WordPunctTokenizer
from dateutil import parser
import re
import numpy as np
from geopy.distance import great_circle
from datetime import datetime

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from llm_handlers.base_handler import BaseLLMHandler

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_closed_metrics.log")

logger = LogManager.get_logger("closed_metrics")

def _preprocess_values(value: Any) -> Union[List[str], str]:
    """
    Preprocess values to handle different input types.
    
    Args:
        value: The input value to preprocess
        
    Returns:
        Union[List[str], str]: The preprocessed value
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    elif isinstance(value, str):
        if ',' in value and not value.startswith('[') and not value.startswith('{'):
            # It might be a comma-separated list
            return _convert_str_to_list(value)
        return value
    else:
        return str(value)

def _convert_str_to_list(str_data: str) -> List[str]:
    """
    Convert string to list.

    Args:
        str_data (str): Data as string

    Returns:
        List[str]: Data as list
    """
    data_list = list(map(lambda x: x.strip(), str_data.split(',')))
    return data_list

def compute_closed_set_metrics(true_labels: List[str], pred_labels: List[str]) -> Tuple[float, float, float]:
    """
    Generate precision, recall, and F1 for predicted
    labels vs. true labels.

    Args:
        true_labels (List[str]): True labels for data
        pred_labels (List[str]): Predicted labels for data
    
    Returns:
        Tuple[float, float, float]: Precision, Recall, F1 score
    """
    tp = len(set(true_labels) & set(pred_labels))
    fp = len(set(pred_labels) - set(true_labels))
    fn = len(set(true_labels) - set(pred_labels))
    
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0.0
        
    if (tp + fn) > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0.0
        
    if (prec + rec) > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    else:
        f1 = 0.0
        
    return prec, rec, f1

def _compute_edit_distance(true_labels: List[str], pred_label: str, use_word_edit_dist: bool = False) -> float:
    """
    Compute string edit distance of best matches from true labels
    to predicted labels.

    Args:
        true_labels (List[str]): True labels for data
        pred_label (str): Predicted label
        use_word_edit_dist (bool): Use word-level edit distance
    
    Returns:
        float: Mean edit distance between predicted label and best-matching true label.
    """
    # Initialize tokenizer if needed
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Error downloading NLTK punkt: {str(e)}")
    
    tokenizer = WordPunctTokenizer()
    
    # Tokenize if word-level edit distance
    if use_word_edit_dist:
        true_labels_processed = list(map(tokenizer.tokenize, true_labels))
        pred_label_processed = tokenizer.tokenize(pred_label)
    else:
        true_labels_processed = true_labels
        pred_label_processed = pred_label
    
    # Calculate edit distances
    pred_label_len = len(pred_label_processed)
    edit_distances = []
    
    for true_label in true_labels_processed:
        true_label_len = len(true_label)
        max_len = max(pred_label_len, true_label_len)
        if max_len == 0:  # Avoid division by zero
            edit_distances.append(0.0)
        else:
            norm_edit_dist = edit_distance(pred_label_processed, true_label) / max_len
            edit_distances.append(norm_edit_dist)
    
    # Return minimum edit distance (best match)
    if edit_distances:
        return min(edit_distances)
    else:
        return 1.0  # Maximum distance if no labels

def _convert_to_number(num_str: str, try_date: bool = True) -> Any:
    """
    Convert string to float, array, or date.

    Args:
        num_str (str): String representing one/more number(s)
        try_date (bool): Whether to try parsing as date
    
    Returns:
        Any: Number, date, or array of numbers
    """
    num = None
    
    # Try parsing as date
    if try_date:
        try:
            num = parser.parse(num_str, fuzzy=True)
            return num
        except Exception:
            pass
    
    # Try parsing as numeric values
    num_vals = re.findall('([\\-\\d\\.]+)', num_str)
    if not num_vals:
        return None
        
    try:
        num_vals = list(map(float, num_vals))
        if len(num_vals) > 1:
            return np.array(num_vals)
        else:
            return num_vals[0]
    except Exception:
        return None

def compute_numerical_error(
    
    true_value: str, 
    pred_value: str, 
    lat_lon: bool = False,
    try_date: bool = False, 
    num_bounds: Optional[List[float]] = None
) -> float:
    """
    Compute numerical error metric between true label and
    predicted label.

    Args:
        true_value (str): True value
        pred_value (str): Predicted value
        lat_lon (bool): Whether values are latitude/longitude coordinates
        num_bounds (List[float]): Min/max bounds for number
    
    Return:
        float: [0,1] normalized numerical error
    """
    true_label_num = _convert_to_number(true_value, try_date)
    pred_label_num = _convert_to_number(pred_value, try_date)
    
    # Return maximum error if conversion failed
    if true_label_num is None or pred_label_num is None:
        return 1.0
    
    # Calculate error based on the type of data
    if lat_lon:
        # For lat/lon, calculate great circle distance
        try:
            # Convert array to tuple if necessary
            if isinstance(true_label_num, np.ndarray):
                true_coords = tuple(true_label_num)
            else:
                true_coords = (true_label_num, 0.0)
            
            if isinstance(pred_label_num, np.ndarray):
                pred_coords = tuple(pred_label_num)
            else:
                pred_coords = (pred_label_num, 0.0)
            
            # Earth's max distance (half circumference in miles)
            MAX_DIST = 24901.461 / 2.0
            geo_err = great_circle(true_coords, pred_coords).miles
            err = geo_err / MAX_DIST
        except Exception:
            logger.warning("Error calculating geographic distance")
            err = 1.0
    else:
        # Calculate regular numerical error
        if isinstance(true_label_num, datetime) and isinstance(pred_label_num, datetime):
            # For dates, calculate time difference
            MAX_YEAR_DIFF = 100
            MAX_SECOND_DIFF = 60 * 60 * 24 * 365.25 * MAX_YEAR_DIFF
            time_diff = abs(true_label_num - pred_label_num)
            err = time_diff.total_seconds() / MAX_SECOND_DIFF
        elif isinstance(true_label_num, np.ndarray) and isinstance(pred_label_num, np.ndarray):
            # For arrays, calculate mean squared error
            if len(true_label_num) == len(pred_label_num):
                err = np.sqrt(((true_label_num - pred_label_num) ** 2.0).sum())
            else:
                err = 1.0
        else:
            # For single numbers
            err = abs(true_label_num - pred_label_num)
            # Apply bounds if provided
            if num_bounds is not None:
                err = (err - num_bounds[0]) / (num_bounds[1] - num_bounds[0])
            else:
                # Default: absolute percent error
                if true_label_num != 0:
                    err = err / abs(true_label_num)
                else:
                    err = 1.0 if pred_label_num != 0 else 0.0
    
    # Cap error at [0.0, 1.0]
    return max(min(err, 1.0), 0.0)

def evaluate_precision(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate precision for closed-set metrics.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        
    Returns:
        float: The precision score
    """
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if not isinstance(pred, list):
        pred = [pred]
        
    precision, _, _ = compute_closed_set_metrics(gt, pred)
    return precision

def evaluate_recall(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate recall for closed-set metrics.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        
    Returns:
        float: The recall score
    """
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if not isinstance(pred, list):
        pred = [pred]
        
    _, recall, _ = compute_closed_set_metrics(gt, pred)
    return recall

def evaluate_f1(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate F1 score for closed-set metrics.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        
    Returns:
        float: The F1 score
    """
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if not isinstance(pred, list):
        pred = [pred]
        
    _, _, f1 = compute_closed_set_metrics(gt, pred)
    return f1

def evaluate_char_edit_distance(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate character-level edit distance.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        
    Returns:
        float: The character-level edit distance
    """
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if isinstance(pred, list) and len(pred) > 0:
        pred = pred[0]  # Use the first prediction for edit distance
        
    return _compute_edit_distance(gt, pred, use_word_edit_dist=False)

def evaluate_word_edit_distance(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate word-level edit distance.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        
    Returns:
        float: The word-level edit distance
    """
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if isinstance(pred, list) and len(pred) > 0:
        pred = pred[0]  # Use the first prediction for edit distance
        
    return _compute_edit_distance(gt, pred, use_word_edit_dist=True)

def evaluate_numerical_error(ground_truth: Any, predicted: Any, num_bounds: Optional[List[float]] = None) -> float:
    """
    Evaluate numerical error.
    
    Args:
        ground_truth: The ground truth value
        predicted: The predicted value
        num_bounds: Optional bounds for the numerical error
        
    Returns:
        float: The numerical error
    """
    return compute_numerical_error(str(ground_truth), str(predicted), try_date=False, lat_lon=False, num_bounds=num_bounds)

def evaluate_geo_distance(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate geographic distance between coordinates.
    
    Args:
        ground_truth: The ground truth coordinates (lat, lon)
        predicted: The predicted coordinates (lat, lon)
        
    Returns:
        float: The geographic distance error
    """
    return compute_numerical_error(str(ground_truth), str(predicted), lat_lon=True)

def evaluate_time_difference(ground_truth: Any, predicted: Any) -> float:
    """
    Evaluate time difference.
    
    Args:
        ground_truth: The ground truth time
        predicted: The predicted time
        
    Returns:
        float: The time difference error
    """
    return compute_numerical_error(str(ground_truth), str(predicted), try_date=True, lat_lon=False)

def _llm_match_prompt_template() -> str:
    """
    Get the default prompt template for LLM matching.
    
    Returns:
        str: The prompt template
    """
    return """
    Determine if the source label is an approximate match for one or more of the target labels.
    The answer should be in JSON format and should reference which of the target labels is matched to the source label.

    Example source label:
    "dog"

    Example target labels:
    ["cat", "big dog"]

    Example match target labels:
    {{'match_target_labels' : ["big dog"]}}

    Source label:
    {}

    Target labels:
    {}

    Match target labels:
    """

def _extract_data_from_str(output_str: str) -> Dict[str, Any]:
    """
    Extract JSON data from raw string.

    Args:
        output_str (str): Output string containing JSON data.
    
    Returns:
        Dict[str, Any]: JSON data
    """
    json_matcher = re.compile(r'\{.+\}')
    output_data_str = json_matcher.search(re.sub(r'\n', '', output_str))
    output_data = {}
    
    if output_data_str is not None:
        output_data_str = output_data_str.group(0)
        try:
            # Use ast.literal_eval instead of eval for safety
            import ast
            output_data = ast.literal_eval(output_data_str)
        except Exception as e:
            logger.warning(f"Error parsing LLM output JSON: {str(e)}")
    
    return output_data

def evaluate_exact_match(ground_truth: Any, predicted: Any) -> bool:
    return 1.0 if (ground_truth == predicted) else 0.0

def _extract_llm_match(
        true_labels: List[str], pred_label: str, 
        llm_client:BaseLLMHandler, 
        match_key: str = 'match_target_labels'
        ) -> List[str]:
    """
    Extract matching labels from LLM response.
    
    Args:
        true_labels (List[str]): List of true labels
        pred_label (str): Predicted label
        match_key (str): Key in the JSON response
        
    Returns:
        List[str]: List of matching true labels
    """
    if not llm_client:
        return []
        
    # Format the prompt
    prompt = _llm_match_prompt_template().format(pred_label, '\n'.join(true_labels))
    
    # Get response from LLM
    try:
        response_text = llm_client.generate_response(prompt)
    except Exception as e:
        logger.warning(f"Error getting LLM response: {str(e)}")
        return []
    
    # Extract and return matches
    json_data = _extract_data_from_str(response_text)
    if match_key in json_data:
        return json_data[match_key]
    else:
        return []

def evaluate_llm_match(ground_truth: Any, predicted: Any, llm_client:BaseLLMHandler) -> float:
    """
    Evaluate LLM-based matching.
    
    Args:
        ground_truth: The ground truth value(s)
        predicted: The predicted value(s)
        llm_client: The LLM client to use to predict LLM match
        
    Returns:
        float: The LLM match score (1.0 if matched, 0.0 if not)
    """
    
    if not llm_client:
        logger.warning("LLM chat engine not available. Cannot evaluate llm_match metric.")
        return 0.0
        
    gt = _preprocess_values(ground_truth)
    pred = _preprocess_values(predicted)
    
    if not isinstance(gt, list):
        gt = [gt]
    if isinstance(pred, list) and len(pred) > 0:
        pred = pred[0]
        
    matches = _extract_llm_match(gt, pred, llm_client=llm_client)
    
    # Return 1.0 if there are matches, 0.0 otherwise
    return 1.0 if matches else 0.0