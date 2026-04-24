import os
import sys
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
import hashlib
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager
from metrics.metrics_evaluator import MetricsEvaluator
from llm_handlers.base_handler import BaseLLMHandler
from metrics.closed_metrics import (
    evaluate_precision, evaluate_recall, evaluate_f1,
    evaluate_char_edit_distance, evaluate_word_edit_distance,
    evaluate_time_difference, evaluate_geo_distance, evaluate_numerical_error,
    compute_numerical_error, compute_closed_set_metrics,
    evaluate_exact_match, evaluate_llm_match
)
from metrics.fuzzy_metrics import (
    soft_precision_recall,
    semantic_similarity_embedding, semantic_similarity_fuzzy,
    abbreviation_similarity_score
)

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_nested_evaluator.log")

logger = LogManager.get_logger(__name__)


class NestedStructureEvaluator:
    """
    A specialized evaluator for comparing nested JSON structures field by field.
    
    This class handles complex nested dictionaries and arrays, flattening them
    into comparable field paths and applying appropriate metrics to each field
    based on field names, value types, and custom configurations.
    """
    
    def __init__(self, metrics_evaluator:MetricsEvaluator=None):
        """
        Initialize the NestedStructureEvaluator.
        
        Args:
            metrics_evaluator: Optional MetricsEvaluator instance for metric calculations.
                              If not provided, basic metrics will be used.
        """
        self.metrics_evaluator = metrics_evaluator
        
        # Define basic metrics if no evaluator provided
        if metrics_evaluator is None:
            self.available_metrics = {
                "semantic_similarity": semantic_similarity_embedding,
                "exact_match": evaluate_exact_match,
                "char_edit_distance": self._simple_char_edit_distance,
                "word_edit_distance": self._simple_word_edit_distance,
                "numerical_error": self._simple_numerical_error,
                "f1": self._simple_f1_score,
                "soft_f1": self.soft_f1
            }
        else:
            self.available_metrics = metrics_evaluator.available_metrics
            
        logger.info(f"NestedStructureEvaluator initialized with {len(self.available_metrics)} available metrics")
    
    def _simple_char_edit_distance(self, expected: Any, predicted: Any) -> float:
        """Simple character-level edit distance using basic string comparison."""
        if expected is None and predicted is None:
            return 1.0
        if expected is None or predicted is None:
            return 0.0
            
        str_expected = str(expected)
        str_predicted = str(predicted)
        
        if len(str_expected) == 0 and len(str_predicted) == 0:
            return 1.0
            
        max_len = max(len(str_expected), len(str_predicted))
        if max_len == 0:
            return 1.0
            
        # Simple character-by-character comparison
        differences = sum(c1 != c2 for c1, c2 in zip(str_expected, str_predicted))
        differences += abs(len(str_expected) - len(str_predicted))
        
        return max(0.0, 1.0 - (differences / max_len))
    
    def _simple_word_edit_distance(self, expected: Any, predicted: Any) -> float:
        """Simple word-level edit distance."""
        if expected is None and predicted is None:
            return 1.0
        if expected is None or predicted is None:
            return 0.0
            
        words_expected = str(expected).split()
        words_predicted = str(predicted).split()
        
        if len(words_expected) == 0 and len(words_predicted) == 0:
            return 1.0
            
        # Simple word overlap calculation
        set_expected = set(words_expected)
        set_predicted = set(words_predicted)
        
        intersection = len(set_expected & set_predicted)
        union = len(set_expected | set_predicted)
        
        return intersection / union if union > 0 else 0.0
    
    def _simple_numerical_error(self, expected: Any, predicted: Any) -> float:
        """Simple numerical error calculation."""
        try:
            num_expected = float(expected) if expected is not None else 0.0
            num_predicted = float(predicted) if predicted is not None else 0.0
            
            if num_expected == 0 and num_predicted == 0:
                return 1.0
            elif num_expected == 0:
                return 0.0
            else:
                error = abs(num_expected - num_predicted) / abs(num_expected)
                return max(0.0, 1.0 - error)
        except (ValueError, TypeError):
            return self.available_metrics['exact_match'](expected, predicted)
    
    def _simple_f1_score(self, expected: Any, predicted: Any) -> float:
        """Simple F1 score for list comparison."""
        if not isinstance(expected, list):
            expected = [expected] if expected is not None else []
        if not isinstance(predicted, list):
            predicted = [predicted] if predicted is not None else []
            
        set_expected = set(str(item) for item in expected)
        set_predicted = set(str(item) for item in predicted)
        
        tp = len(set_expected & set_predicted)
        fp = len(set_predicted - set_expected)
        fn = len(set_expected - set_predicted)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def soft_f1(self, expected: List[str], predicted: List[str]) -> float:
        """Get only F1 score, using cached computation if available."""
        _, _, f1 = soft_precision_recall(
                expected, predicted, 
                embedding_model = "all-MiniLM-L6-v2"
            )
        return f1
    
    def _safe_flatten_nested_dict(
        self, 
        data: Any, 
        parent_key: str = '', 
        sep: str = '.', 
        list_handling: str = 'indexed'
    ) -> Dict[str, Any]:
        """
        Safely flatten a nested dictionary with proper null checks.
        
        Args:
            data: The data to flatten (can be dict, list, or any value)
            parent_key (str): The parent key for the current level
            sep (str): Separator for nested keys
            list_handling (str): How to handle lists
                
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items = []
        
        # Handle None data
        if data is None:
            if parent_key:
                items.append((parent_key, None))
            return dict(items)
        
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    nested_result = self._safe_flatten_nested_dict(v, new_key, sep=sep, list_handling=list_handling)
                    if nested_result:  # Check if result is not empty
                        items.extend(nested_result.items())
                elif isinstance(v, list):
                    if list_handling == 'indexed':
                        # Original behavior: create indexed keys
                        for i, item in enumerate(v):
                            list_key = f"{new_key}[{i}]"
                            if isinstance(item, dict):
                                nested_result = self._safe_flatten_nested_dict(item, list_key, sep=sep, list_handling=list_handling)
                                if nested_result:  # Check if result is not empty
                                    items.extend(nested_result.items())
                            else:
                                items.append((list_key, item))
                    
                    elif list_handling == 'merged':
                        # Merge all list items - collect all values for each field
                        if v:  # Only process non-empty lists
                            merged_dict = {}
                            for item in v:
                                if isinstance(item, dict):
                                    flattened_item = self._safe_flatten_nested_dict(item, '', sep=sep, list_handling=list_handling)
                                    if flattened_item:  # Check if result is not empty
                                        for sub_key, sub_value in flattened_item.items():
                                            full_key = f"{new_key}{sep}{sub_key}"
                                            if full_key not in merged_dict:
                                                merged_dict[full_key] = []
                                            merged_dict[full_key].append(sub_value)
                                else:
                                    if new_key not in merged_dict:
                                        merged_dict[new_key] = []
                                    merged_dict[new_key].append(item)
                            items.extend(merged_dict.items())
                        else:
                            items.append((new_key, []))
                    
                    elif list_handling == 'union_keys':
                        # Create union of all possible keys, pad missing values with None
                        if v:  # Only process non-empty lists
                            # First pass: collect all possible keys
                            all_keys = set()
                            flattened_items = []
                            
                            for item in v:
                                if isinstance(item, dict):
                                    flattened_item = self._safe_flatten_nested_dict(item, '', sep=sep, list_handling='indexed')
                                    if flattened_item:  # Check if result is not empty
                                        flattened_items.append(flattened_item)
                                        all_keys.update(flattened_item.keys())
                                    else:
                                        flattened_items.append({})
                                else:
                                    flattened_items.append({new_key: item})
                                    all_keys.add(new_key)
                            
                            # Second pass: create indexed entries for all keys
                            max_index = len(v) - 1
                            for key in sorted(all_keys):
                                for i in range(max_index + 1):
                                    indexed_key = f"{new_key}[{i}]{sep}{key}" if key != new_key else f"{new_key}[{i}]"
                                    
                                    if i < len(flattened_items) and flattened_items[i]:
                                        value = flattened_items[i].get(key, None)
                                    else:
                                        value = None
                                    
                                    items.append((indexed_key, value))
                        else:
                            items.append((new_key, []))
                    
                else:
                    items.append((new_key, v))
        else:
            # Handle non-dict values
            if parent_key:
                items.append((parent_key, data))
        
        return dict(items)
    
    def flatten_nested_dict(
        self, 
        data: Dict[str, Any], 
        parent_key: str = '', 
        sep: str = '.', 
        list_handling: str = 'indexed'
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary into a flat dictionary with dot-separated keys.
        This is a wrapper around the safe implementation.
        
        Args:
            data (Dict[str, Any]): The nested dictionary to flatten
            parent_key (str): The parent key for the current level
            sep (str): Separator for nested keys
            list_handling (str): How to handle lists
                
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        try:
            return self._safe_flatten_nested_dict(data, parent_key, sep, list_handling)
        except Exception as e:
            logger.error(f"Error flattening nested dict: {str(e)}")
            # Return a safe fallback
            if data is None:
                return {}
            elif isinstance(data, dict):
                # Simple fallback: just flatten one level
                result = {}
                for k, v in data.items():
                    key = f"{parent_key}{sep}{k}" if parent_key else k
                    result[key] = v
                return result
            else:
                return {parent_key: data} if parent_key else {}
        
    def _align_list_structures(
        self, 
        expected_flat: Dict[str, Any], 
        predicted_flat: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Align flattened structures when lists have different lengths.
        
        Args:
            expected_flat (Dict[str, Any]): Flattened expected structure
            predicted_flat (Dict[str, Any]): Flattened predicted structure
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Aligned expected and predicted structures
        """
        aligned_expected = expected_flat.copy()
        aligned_predicted = predicted_flat.copy()
        
        # Find all list-based keys (containing [index])
        expected_list_keys = {}  # base_key -> {index -> full_key}
        predicted_list_keys = {}
        
        # Parse expected keys
        for key in expected_flat.keys():
            if '[' in key and ']' in key:
                # Extract base key and index
                bracket_start = key.find('[')
                bracket_end = key.find(']')
                base_key = key[:bracket_start]
                try:
                    index = int(key[bracket_start+1:bracket_end])
                    suffix = key[bracket_end+1:]
                    
                    if base_key not in expected_list_keys:
                        expected_list_keys[base_key] = {}
                    expected_list_keys[base_key][index] = key
                except ValueError:
                    continue
        
        # Parse predicted keys
        for key in predicted_flat.keys():
            if '[' in key and ']' in key:
                bracket_start = key.find('[')
                bracket_end = key.find(']')
                base_key = key[:bracket_start]
                try:
                    index = int(key[bracket_start+1:bracket_end])
                    suffix = key[bracket_end+1:]
                    
                    if base_key not in predicted_list_keys:
                        predicted_list_keys[base_key] = {}
                    predicted_list_keys[base_key][index] = key
                except ValueError:
                    continue
        
        # Align structures for each list base key
        for base_key in set(expected_list_keys.keys()) | set(predicted_list_keys.keys()):
            expected_indices = set(expected_list_keys.get(base_key, {}).keys())
            predicted_indices = set(predicted_list_keys.get(base_key, {}).keys())
            
            max_expected_index = max(expected_indices) if expected_indices else -1
            max_predicted_index = max(predicted_indices) if predicted_indices else -1
            max_index = max(max_expected_index, max_predicted_index)
            
            # Get all field suffixes for this base key
            expected_suffixes = set()
            predicted_suffixes = set()
            
            for key in expected_flat.keys():
                if key.startswith(f"{base_key}[") and ']' in key:
                    bracket_end = key.find(']')
                    suffix = key[bracket_end+1:]
                    expected_suffixes.add(suffix)
            
            for key in predicted_flat.keys():
                if key.startswith(f"{base_key}[") and ']' in key:
                    bracket_end = key.find(']')
                    suffix = key[bracket_end+1:]
                    predicted_suffixes.add(suffix)
            
            all_suffixes = expected_suffixes | predicted_suffixes
            
            # Ensure all combinations exist
            for index in range(max_index + 1):
                for suffix in all_suffixes:
                    expected_key = f"{base_key}[{index}]{suffix}"
                    predicted_key = f"{base_key}[{index}]{suffix}"
                    
                    if expected_key not in aligned_expected:
                        aligned_expected[expected_key] = None
                    if predicted_key not in aligned_predicted:
                        aligned_predicted[predicted_key] = None
        
        return aligned_expected, aligned_predicted
    
    def get_field_metric_type(self, field_path: str, value: Any) -> str:
        """
        Determine the appropriate metric type for a field based on its path and value.
        
        Args:
            field_path (str): The dot-separated path to the field
            value (Any): The field value
            
        Returns:
            str: The suggested metric type
        """
        # Convert value to string for analysis
        str_value = str(value).lower() if value is not None else ""
        
        # Field-specific rules based on common patterns
        field_lower = field_path.lower()

        if isinstance(value, list):
            # Email fields
            if 'email' in field_lower:
                return 'f1'
            else:
                return 'soft_f1'
        
        # ID fields
        if field_lower.endswith('id') or 'uuid' in field_lower or 'docket' in field_lower:
            return 'exact_match'
        
        # Date/Time fields
        if any(keyword in field_lower for keyword in ['date', 'time', 'startdate', 'enddate']):
            if self.metrics_evaluator and 'time_difference' in self.available_metrics:
                if str_value and str_value != 'null' and str_value != '':
                    return 'time_difference'
            return 'exact_match'
        
        # Numeric fields (phone, zip)
        if any(keyword in field_lower for keyword in ['phone', 'zip', 'zipcode']):
            return 'numerical_error'
        
        # Location/Address fields
        if any(keyword in field_lower for keyword in ['street', 'city', 'state', 'location', 'address']):
            return 'semantic_similarity'
        
        # Person/Position/Department fields (longer text)
        if any(keyword in field_lower for keyword in ['person', 'position', 'department', 'notes', 'description']):
            return 'semantic_similarity'
        
        # Type fields (categorical)
        if 'type' in field_lower:
            return 'exact_match'
        
        # Default based on value characteristics
        if isinstance(value, (int, float)):
            return 'numerical_error'
        elif isinstance(value, str):
            return 'semantic_similarity'
        elif isinstance(value, list):
            return 'f1'  # For list comparison
        else:
            return 'exact_match'
    
    def evaluate_nested_structure(
        self,
        response_entry: Dict[str, Any],
        field_metric_config: Optional[Dict[str, str]] = None,
        auto_detect_metrics: bool = True,
        list_handling: str = 'indexed',
        align_lists: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate metrics for nested JSON structures by comparing answer_expected and answer_predicted.
        
        Args:
            response_entry (Dict[str, Any]): Single response entry containing answer_expected and answer_predicted
            field_metric_config (Optional[Dict[str, str]]): Mapping of field paths to metric types
                Format: {"Notice.ID": "exact_match", "Comments.StartDate.Date": "time_difference"}
            auto_detect_metrics (bool): Whether to auto-detect metrics for fields not in config
            list_handling (str): How to handle lists:
                - 'indexed': Create indexed keys like "field[0].subfield", "field[1].subfield"
                - 'merged': Merge all list items into field names like "field.subfield" (all values)
                - 'union_keys': Create union of all possible keys across list items
            align_lists (bool): Whether to align list structures when they have different lengths
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict[str, Any]: Dictionary containing field-level metric scores and overall statistics
            
        Raises:
            ValueError: If required fields are missing
        """
        if "answer_expected" not in response_entry or "answer_predicted" not in response_entry:
            raise ValueError("Response entry must contain 'answer_expected' and 'answer_predicted' fields")
        
        expected = response_entry["answer_expected"]
        predicted = response_entry["answer_predicted"]
        
        # Flatten both structures
        expected_flat = self.flatten_nested_dict(expected, list_handling=list_handling)
        predicted_flat = self.flatten_nested_dict(predicted, list_handling=list_handling)
        
        # Align list structures if requested and using indexed mode
        if align_lists and list_handling == 'indexed':
            expected_flat, predicted_flat = self._align_list_structures(expected_flat, predicted_flat)
        
        # Get all unique from expected only
        all_fields = set(expected_flat.keys())
        
        logger.info(f"Evaluating {len(all_fields)} fields in nested structure (list_handling='{list_handling}', align_lists={align_lists})")
        
        field_results = {}
        metric_type_counts = {}
        
        for field_path in sorted(all_fields):
            expected_value = expected_flat.get(field_path)

            if isinstance(expected_value, list):
                if len(expected_value) == 0:
                    predicted_value = predicted_flat.get(field_path, [])
                else:
                    predicted_value = predicted_flat.get(field_path, [''])
                # Handle string predicted values for list expected
                if not isinstance(predicted_value, list):
                    predicted_value = [predicted_value]

            elif isinstance(expected_value, str):
                predicted_value = predicted_flat.get(field_path, "")
                # Handle list predicted values for non-list expected
                if isinstance(predicted_value, list):
                    expected_value = [expected_value]

            else:
                predicted_value = predicted_flat.get(field_path, None)
                
            
            logger.info(f"Evaluating field: {field_path}: expected={expected_value} | predicted={predicted_value}")
            
            # Determine metric type
            if field_metric_config and field_path in field_metric_config:
                metric_type = field_metric_config[field_path]
            elif auto_detect_metrics:
                metric_type = self.get_field_metric_type(field_path, expected_value)
            else:
                metric_type = 'exact_match'  # Default fallback
            
            # Track metric type usage
            metric_type_counts[metric_type] = metric_type_counts.get(metric_type, 0) + 1
            
            # Evaluate the metric
            try:
                if metric_type in self.available_metrics:
                    if self.metrics_evaluator:
                        score = self.available_metrics[metric_type](expected_value, predicted_value, **kwargs)
                    else:
                        score = self.available_metrics[metric_type](expected_value, predicted_value)
                else:
                    logger.warning(f"Unknown metric type '{metric_type}' for field '{field_path}'. Using exact_match.")
                    score = 1.0 if expected_value == predicted_value else 0.0
                    
                field_results[field_path] = {
                    "expected": expected_value,
                    "predicted": predicted_value,
                    "metric_type": metric_type,
                    "score": score
                }
                
                logger.debug(f"Field '{field_path}': {metric_type} = {score}")
                
            except Exception as e:
                logger.warning(f"Error evaluating field '{field_path}' with metric '{metric_type}': {str(e)}")
                field_results[field_path] = {
                    "expected": expected_value,
                    "predicted": predicted_value,
                    "metric_type": metric_type,
                    "score": float('nan'),
                    "error": str(e)
                }
        
        # Calculate overall statistics
        all_scores = [result["score"] for result in field_results.values() 
                      if not np.isnan(result["score"])]
        
        overall_stats = {
            "total_fields": len(all_fields),
            "evaluated_fields": len(all_scores),
            "failed_evaluations": len(all_fields) - len(all_scores),
            "average_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "metric_type_distribution": metric_type_counts,
            "list_handling_mode": list_handling,
            "list_alignment_enabled": align_lists
        }
        
        # Group scores by metric type
        scores_by_metric = {}
        for result in field_results.values():
            metric_type = result["metric_type"]
            score = result["score"]
            if not np.isnan(score):
                if metric_type not in scores_by_metric:
                    scores_by_metric[metric_type] = []
                scores_by_metric[metric_type].append(score)
        
        # Calculate average scores by metric type
        metric_averages = {
            metric_type: float(np.mean(scores))
            for metric_type, scores in scores_by_metric.items()
        }
        
        result = {
            "field_results": field_results,
            "overall_statistics": overall_stats,
            "metric_type_averages": metric_averages
        }
        
        logger.info(f"Nested structure evaluation completed. Average score: {overall_stats['average_score']:.3f}")
        
        return result
    
    def batch_evaluate_nested_structures(
        self,
        response_path: str,
        field_metric_config: Optional[Dict[str, str]] = None,
        auto_detect_metrics: bool = True,
        list_handling: str = 'indexed',
        align_lists: bool = True,
        output_path: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple nested structures from a JSON file.
        
        Args:
            response_path (str): Path to JSON file containing responses
            field_metric_config (Optional[Dict[str, str]]): Mapping of field paths to metric types
            auto_detect_metrics (bool): Whether to auto-detect metrics for fields not in config
            list_handling (str): How to handle lists ('indexed', 'merged', 'union_keys')
            align_lists (bool): Whether to align list structures when they have different lengths
            output_path (Optional[str]): Optional path to save results
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            List[Dict[str, Any]]: List of evaluation results for each response
        """
        # Load responses
        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Response file not found: {response_path}")
            
        try:
            with open(response_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both list format and dict format
            if isinstance(data, dict):
                responses = list(data.values())
            elif isinstance(data, list):
                responses = data
            else:
                raise ValueError("JSON file must contain a list or dictionary of responses")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {response_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading responses: {str(e)}")
            raise
        
        logger.info(f"Batch evaluating {len(responses)} nested structures")
        
        results = []
        for i, response in enumerate(responses):
            try:
                result = self.evaluate_nested_structure(
                    response, 
                    field_metric_config=field_metric_config,
                    auto_detect_metrics=auto_detect_metrics,
                    list_handling=list_handling,
                    align_lists=align_lists,
                    **kwargs
                )
                result["response_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error evaluating response {i}: {str(e)}")
                results.append({
                    "response_index": i,
                    "error": str(e),
                    "field_results": {},
                    "overall_statistics": {
                        "total_fields": 0,
                        "evaluated_fields": 0,
                        "failed_evaluations": 1,
                        "average_score": 0.0,
                        "metric_type_distribution": {}
                    },
                    "metric_type_averages": {}
                })
        
        # Save results if output path provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved nested structure evaluation results to {output_path}")
        
        return results
    
    def load_field_metric_config(self, config_path: str) -> Dict[str, str]:
        """
        Load field-metric configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            Dict[str, str]: Mapping of field paths to metric types
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            logger.info(f"Loaded field-metric configuration for {len(config)} fields")
            return config
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON configuration file: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise


# Example usage
def example_usage():
    # Example nested structure
    sample_response = {
        "answer_expected": {
            "Notice": {
                "ID": "UUID#2024-26296",
                "Type": "Notice of Availability"
            },
            "Comments": {
                "Docket": "2024-0002",
                "StartDate": {
                    "Date": "",
                    "Time": ""
                },
                "EndDate": {
                    "Date": "",
                    "Time": ""
                },
                "PhysicalMailing": [
                    {
                        "Street": "1301 A Street, Suite 610",
                        "City": "Tacoma",
                        "State": "WA",
                        "Zip": "98402",
                        "Person": "Patrick Manning",
                        "Email": ["lyndenlpoe@gsa.gov", "sumaslpoe@gsa.gov"]
                    },
                    {
                        "Street": "1805 B Street, Suite 602",
                        "City": "Seattle",
                        "State": "WA",
                        "Zip": "98133",
                        "Person": "John Doe",
                        "Email": ["john@gsa.gov", "doe@gsa.gov"]
                    }
                ]
            }
        },
        "answer_predicted": {
            "Notice": {
                "ID": "2024-0002",
                "Type": "Notice of Availability"
            },
            "Comments": {
                "Docket": "2024-0002",
                "StartDate": {
                    "Date": "November 14, 2024",
                    "Time": None
                },
                "EndDate": {
                    "Date": "December 16, 2024",
                    "Time": None
                },
                "PhysicalMailing": [
                    {
                        "Street": "1301 A Street, Suite 610",
                        "City": "Tacoma",
                        "State": "WA",
                        "Zip": "98402",
                        "Person": "Patrick Manning",
                        "Email": ["lyndenlpoe@gsa.gov", "sumaslpoe@gsa.gov"]
                    }
                ]
            }
        }
    }
    
    # Initialize evaluator
    evaluator = NestedStructureEvaluator()
    
    print("=== INDEXED MODE WITH ALIGNMENT ===")
    result = evaluator.evaluate_nested_structure(
        sample_response, 
        list_handling='indexed', 
        align_lists=True
    )
    print(f"Overall Score: {result['overall_statistics']['average_score']:.3f}")
    print(f"Total Fields: {result['overall_statistics']['total_fields']}")
    print("\nField Results:")
    for field, details in result['field_results'].items():
        if 'PhysicalMailing' in field:
            print(f"  {field}: {details['score']:.3f} | Expected: {details['expected']} | Predicted: {details['predicted']}")
    
    print("\n=== MERGED MODE ===")
    result_merged = evaluator.evaluate_nested_structure(
        sample_response, 
        list_handling='merged'
    )
    print(f"Overall Score: {result_merged['overall_statistics']['average_score']:.3f}")
    print("\nField Results (merged lists):")
    for field, details in result_merged['field_results'].items():
        if 'PhysicalMailing' in field:
            print(f"  {field}: {details['score']:.3f} | Expected: {details['expected']} | Predicted: {details['predicted']}")
    
    print("\n=== UNION KEYS MODE ===")
    result_union = evaluator.evaluate_nested_structure(
        sample_response, 
        list_handling='union_keys'
    )
    print(f"Overall Score: {result_union['overall_statistics']['average_score']:.3f}")
    print("\nField Results (union keys):")
    for field, details in result_union['field_results'].items():
        if 'PhysicalMailing' in field:
            print(f"  {field}: {details['score']:.3f} | Expected: {details['expected']} | Predicted: {details['predicted']}")


if __name__ == "__main__":
    
    # Response and score file paths
    response_file = "results/information_extraction/fedreg_gpt4/responses.json"
    scores_file = "results/information_extraction/fedreg_gpt4/scores.json"

    # Initialize evaluator
    evaluator = NestedStructureEvaluator()

    # Load responses
    with open(response_file, 'r', encoding='utf-8') as f:
        responses = json.load(f)

    evaluator.evaluate_nested_structure(responses['sie_1'], list_handling='merged')