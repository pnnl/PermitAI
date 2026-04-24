from collections import defaultdict
from math import isclose
import os
import sys
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
import hashlib


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from metrics.closed_metrics import (
    evaluate_precision, evaluate_recall, evaluate_f1,
    evaluate_char_edit_distance, evaluate_word_edit_distance,
    evaluate_time_difference, evaluate_geo_distance, evaluate_numerical_error,
    compute_numerical_error, compute_closed_set_metrics,
    evaluate_exact_match
)
from metrics.fuzzy_metrics import (
    soft_precision_recall,
    semantic_similarity_embedding, semantic_similarity_fuzzy,
    abbreviation_similarity_score
)

# Set up logging
if __name__ == "__main__":
    LogManager.initialize("logs/test_metric_evaluator.log")

logger = LogManager.get_logger(__name__)



class MetricsEvaluator:
    """
    Abstract base class for evaluating metrics between ground truth and predicted values.
    
    This class provides methods to evaluate different types of metrics:
    - Closed-set metrics (precision, recall, F1)
    - Edit distance metrics (character-level or word-level)
    - Numerical metrics (relative error, time difference, geographic distance)
    - LLM-based matching metrics
    
    It also supports batch evaluation of multiple responses against multiple metrics.
    """
    
    def __init__(self):
        """
        Initialize the MetricsEvaluator.
        """

        # Initialize cache
        self._cache = {}
        
        # Register the available metrics
        self.available_metrics = {
            "precision": evaluate_precision,
            "recall": evaluate_recall,
            "f1": evaluate_f1,
            "char_edit_distance": evaluate_char_edit_distance,
            "word_edit_distance": evaluate_word_edit_distance,
            "numerical_error": evaluate_numerical_error,
            "geo_distance": evaluate_geo_distance,
            "time_difference": evaluate_time_difference,
            "exact_match": evaluate_exact_match,
            "semantic_similarity_embedding": semantic_similarity_embedding,
            "semantic_similarity_fuzzy": semantic_similarity_fuzzy,
            "abbreviation_similarity": abbreviation_similarity_score,
            "soft_precision": self.soft_precision,
            "soft_recall": self.soft_recall,
            "soft_f1": self.soft_f1,
        }
        
        logger.info(f"MetricsEvaluator initialized with {len(self.available_metrics)} available metrics")
        
    def _create_cache_key(self, expected: List[str], possible: List[str], 
                         embedding_model: Optional[str] = None,
                         use_abbreviation_matching: bool = False) -> str:
        """Create a unique cache key for the given parameters."""
        # Create a string representation of the parameters
        key_data = {
            'expected': tuple(expected),
            'possible': tuple(possible), 
            'embedding_model': embedding_model,
            'use_abbreviation_matching': use_abbreviation_matching
        }
        key_str = str(key_data)
        
        # Use hash for shorter key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _compute_and_cache(self, expected: List[str], possible: List[str],
                          embedding_model: Optional[str] = None,
                          normalize: Optional[Callable[[str], str]] = None,
                          use_abbreviation_matching: bool = False) -> Tuple[float, float, float]:
        """Compute metrics and cache the result."""
        cache_key = self._create_cache_key(expected, possible, embedding_model, use_abbreviation_matching)
        
        if cache_key not in self._cache:
            
            result = soft_precision_recall(
                expected, possible, embedding_model, normalize, use_abbreviation_matching
            )
            self._cache[cache_key] = result
            logger.info(f"Computed and cached metrics for key: {cache_key[:8]}...")
        else:
            logger.info(f"Retrieved cached metrics for key: {cache_key[:8]}...")
            
        return self._cache[cache_key]
    
    def soft_precision(self, expected: List[str], predicted: List[str], **kwargs) -> float:
        """Get only precision score, using cached computation if available."""
        precision, _, _ = self._compute_and_cache(expected, predicted, **kwargs)
        return precision
    
    def soft_recall(self, expected: List[str], predicted: List[str], **kwargs) -> float:
        """Get only recall score, using cached computation if available."""
        _, recall, _ = self._compute_and_cache(expected, predicted, **kwargs)
        return recall
    
    def soft_f1(self, expected: List[str], predicted: List[str], **kwargs) -> float:
        """Get only F1 score, using cached computation if available."""
        _, _, f1 = self._compute_and_cache(expected, predicted, **kwargs)
        return f1
    
    def evaluate(self, ground_truth: Any, predicted: Any, metric: str, **kwargs) -> float:
        """
        Evaluate a single metric between ground truth and predicted values.
        
        Args:
            ground_truth: The ground truth value(s)
            predicted: The predicted value(s)
            metric: The name of the metric to evaluate
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            float: The metric score
            
        Raises:
            ValueError: If the metric is not available
        """
        if metric not in self.available_metrics:
            raise ValueError(f"Metric '{metric}' not available. Available metrics: {list(self.available_metrics.keys())}")
        
        logger.info(f"Evaluating metric: {metric}")
        return self.available_metrics[metric](ground_truth, predicted, **kwargs)
    
    def load_existing_scores(self, output_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load existing computed scores from output file.
        
        Args:
            output_path: Path to the output scores file
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping entry_id to scores
        """
        if not os.path.exists(output_path):
            logger.info(f"No existing scores file found at {output_path}")
            return {}
            
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                scores_data = json.load(f)
                
            logger.info(f"Loaded existing scores for {len(scores_data)} entries from {output_path}")

            # Convert the list of scores to a dictionary for easy lookup
            scores_data = {entry['entry_id']: entry['scores'] for entry in scores_data}
            return scores_data
            
        except Exception as e:
            logger.warning(f"Error loading existing scores: {str(e)}. Starting fresh evaluation.")
            return {}
    
    def load_responses(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load responses from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing responses
            
        Returns:
            List[Dict[str, Any]]: List of response entries
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file is not a list or doesn't contain required fields
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                dict_responses = json.load(f)
            responses = list(dict_responses.values())
                
            if not isinstance(responses, list):
                raise ValueError("JSON file must contain a list of response entries")
                
            # Check that each response has required fields
            for i, response in enumerate(responses):
                if not isinstance(response, dict):
                    raise ValueError(f"Response entry at index {i} is not a dictionary")
                    
                if "answer_expected" not in response:
                    raise ValueError(f"Response entry at index {i} is missing 'answer_expected' field")
                    
                if "answer_predicted" not in response:
                    raise ValueError(f"Response entry at index {i} is missing 'answer_predicted' field")
                    
            logger.info(f"Loaded {len(responses)} responses from {json_path}")
            return responses
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {json_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading responses: {str(e)}")
            raise

    def load_metrics_and_weights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        # Retrieve the metrics and weights dictionaries
        metrics_dict = config["metrics"]
        if not isinstance(metrics_dict, dict):
            if isinstance(metrics_dict, list):
                logger.info(f"Metrics must be a dictionary of metric names to keyword argument dictionaries, got list")
                # Convert list to dict with empty kwargs
                metrics_dict = {metric: {} for metric in metrics_dict}
            else:
                raise ValueError("Metrics must be a dictionary of metric names to keyword argument dictionaries")
            
        # Validate that all metrics are available
        for metric in metrics_dict:
            if metric not in self.available_metrics:
                raise ValueError(f"Unknown metric '{metric}'. Available metrics: {list(self.available_metrics.keys())}")
        
        # Check for weights and validate length
        weights_dict = None
        if "weights" in config:
            weights_dict = config["weights"]
            if not isinstance(weights_dict, dict):
                raise ValueError(f"Weights must be a dictionary of metric names to weights")
                
            if set(weights_dict.keys()) == set(metrics_dict.keys()):
                raise ValueError(f"Weights ({len(weights_dict)}) must have same keys as metrics ({len(metrics_dict)})")
                
            # Validate that all weights are numeric
            for metric, weight in enumerate(weights_dict):
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight for metric {metric} must be numeric, got {type(weight)}")
                    
        else:
            logger.info(f"Loaded {len(metrics_dict)} metrics without weights")
        
        return {
            "metrics": metrics_dict,
            "weights": weights_dict
            }
    
    def load_field_metrics_config(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load field-specific metrics configuration from a JSON file.
        
        Args:
            config_path (str): Path to the JSON configuration file containing field-metric mappings
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping field names to their config (metrics with keyword arguments and optional weights)
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the JSON file format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Metrics configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            if not isinstance(config_data, dict):
                raise ValueError("Configuration file must contain a dictionary")
                
            # Extract metrics and weights from the nested structure
            field_configs = {}
            for field_name, field_config in config_data.items():
                if not isinstance(field_config, dict) or "metrics" not in field_config:
                    raise ValueError(f"Field '{field_name}' must have a 'metrics' key with a list of metrics")

                # Load metrics and weights for this field
                logger.info(f"Loading metrics/weights for field: {field_name}")       
                field_configs[field_name] = self.load_metrics_and_weights(field_config)
                
            logger.info(f"Loaded metrics configuration for {len(field_configs)} fields from {config_path}")
            return field_configs
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading metrics configuration: {str(e)}")
            raise

    def batch_evaluate(
        self, 
        response_path: str, 
        config: Dict[str, Any],
        output_path: Optional[str] = None,
        field_name_key: str = "field_name",
        continue_from_previous: bool = True,
        nepabench_directory:str = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate metrics for a list of responses based on configuration.
        
        Args:
            response_path (str): Path to JSON file containing responses
            config (Dict[str, Any]): Configuration dictionary with either:
                - "metrics_config": Path to JSON file with field-specific metrics
                - "metrics": List of metrics to apply to all responses
                - "weights": Optional list of weights (must match metrics length)
            output_path (Optional[str]): Optional path to save the evaluation results
            field_name_key (str): Key in response entries that contains the field name (only used with metrics_config)
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            List[Dict[str, Any]]: List of response entries with added metric scores
            
        Raises:
            ValueError: If config format is invalid or field not found in config
            FileNotFoundError: If response or config files don't exist
        """
        # Load responses
        all_responses = self.load_responses(response_path)
        
        # Determine configuration type
        if "metrics_config" in config:
            # Field-specific metrics from JSON file
            if nepabench_directory:
                config["metrics_config"] = os.path.join(nepabench_directory, config["metrics_config"])
            field_configs = self.load_field_metrics_config(config["metrics_config"])
            use_field_specific = True
            logger.info(f"Using field-specific metrics configuration from {config['metrics_config']}")
            
        elif "metrics" in config:
            # Same metrics for all responses
            metrics_config = self.load_metrics_and_weights(config)
            metrics_for_response = metrics_config["metrics"]
            weights_for_response = metrics_config["weights"]
            use_field_specific = False
            logger.info(f"Using global metrics configuration with {len(metrics_for_response)} metrics")
            
        else:
            logger.error("Config must contain either 'metrics_config' or 'metrics' key")
            raise ValueError("Config must contain either 'metrics_config' or 'metrics' key")
        
        # Load existing scores if continuing from previous
        existing_scores = {}
        responses_to_evaluate = all_responses
        results = []
        
        if continue_from_previous:
            existing_scores = self.load_existing_scores(output_path)
            
            # Filter out responses that already have scores
            responses_to_evaluate = []
            for response in all_responses:
                entry_id = response['entry_id']
                if entry_id not in existing_scores:
                    responses_to_evaluate.append(response)
                else:
                    if use_field_specific:
                        field_name = response[field_name_key]
                        metrics_to_compute = field_configs[field_name]["metrics"]
                        if not all(metric in existing_scores[entry_id] for metric in metrics_to_compute):
                            responses_to_evaluate.append(response)
                        else:
                            results.append({**response, "scores": existing_scores[entry_id]})
                    else:
                        if not all(metric in existing_scores[entry_id] for metric in metrics_for_response):
                            responses_to_evaluate.append(response)
                        else:
                            results.append({**response, "scores": existing_scores[entry_id]})
            
            logger.info(f"Found {len(existing_scores)} existing scores")
            logger.info(f"Need to evaluate {len(responses_to_evaluate)} remaining responses")
        
        logger.info(f"Evaluating {len(responses_to_evaluate)} responses")
        
        for i, response in enumerate(responses_to_evaluate):
            # Copy the original response
            result = response.copy()
            result["scores"] = {}
            
            # Get ground truth and predicted values
            ground_truth = response["answer_expected"]
            predicted = response["answer_predicted"]
            
            if use_field_specific:
                # Field-specific metrics
                if field_name_key not in response:
                    logger.warning(f"Response {i} missing '{field_name_key}' key. Skipping.")
                    results.append(result)
                    continue
                    
                field_name = response[field_name_key]
                
                if field_name not in field_configs:
                    logger.warning(f"No metrics configuration found for field '{field_name}' in response {i}. Skipping.")
                    results.append(result)
                    continue
                    
                field_config = field_configs[field_name]
                metrics_for_response = field_config["metrics"]
                weights_for_response = field_config["weights"]
            
            # Evaluate each metric for this response
            metric_scores = {}
            for metric, metric_kwargs in metrics_for_response.items():
                try:
                    metric_score = self.available_metrics[metric](
                        ground_truth, predicted, 
                        **metric_kwargs
                        )
                    
                    result["scores"][metric] = metric_score
                    metric_scores[metric] = metric_score
                    
                    if use_field_specific:
                        logger.debug(f"Field '{field_name}', metric '{metric}': {metric_score}")
                    else:
                        logger.debug(f"Response {i}, metric '{metric}': {metric_score}")
                        
                except Exception as e:
                    if use_field_specific:
                        logger.warning(f"Error evaluating metric '{metric}' for field '{field_name}' in response {i}: {str(e)}")
                    else:
                        logger.warning(f"Error evaluating metric '{metric}' for response {i}: {str(e)}")
                    metric_score = float('nan')
                    result["scores"][metric] = metric_score
                    metric_scores[metric] = metric_score
            
            # Calculate weighted or simple average/no-avergae computation
            if weights_for_response is not None:
                try:
                    # Filter out NaN values and corresponding weights
                    valid_scores = []
                    valid_weights = []
                    for metric, weight in weights_for_response.items():
                        score = metric_scores[metric]
                        if not np.isnan(score):
                            valid_scores.append(score)
                            valid_weights.append(weight)
                    
                    if valid_scores:
                        # Calculate weighted average
                        weighted_avg = np.average(valid_scores, weights=valid_weights)
                        result["aggregate"] = float(weighted_avg)
                        
                        if use_field_specific:
                            logger.debug(f"Field '{field_name}': Weighted average score = {weighted_avg}")
                        else:
                            logger.debug(f"Response {i}: Weighted average score = {weighted_avg}")
                    else:
                        result["aggregate"] = float('nan')
                        logger.warning(f"Response {i}: No valid scores for weighted average")
                        
                except Exception as e:
                    logger.warning(f"Error calculating weighted average for response {i}: {str(e)}")
                    result["weighted_average_score"] = float('nan')
            else:
                logger.info("No weights provided for aggregate computation")
                    
            results.append(result)
            
        # Save results if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved {len(results)} evaluation results to {output_path}")
            
        return results

    def compute_score_by_metric_type(
        self, true_val, pred_val, 
        metric_type,
    ):
        """
        Compute single score based on metric type.

        Args:
            true_val: True field value
            pred_val: Predicted field value
            metric_type: Metric to use for prediction
            
        Returns:
            float: Score
        """
        score = 0
        if(metric_type == 'exact_match'):
            if(true_val == pred_val):
                score = 1
        elif(metric_type == 'closed_set'):
            prec, rec, f1 = compute_closed_set_metrics(
                true_val, pred_val
            )
            ## only care about f1
            score = f1
        ## TODO: enable semantic similarity, judge LLM
        elif(metric_type == 'open_set'):
            edit_dist = evaluate_char_edit_distance(
                true_val, pred_val
            )
            ## higher = better
            edit_score = 1 - edit_dist
            score = edit_score
        elif(metric_type == 'numeric'):
            num_err = compute_numerical_error(
                true_val, pred_val
            )
            ## higher = better
            num_score = 1 - num_err
            score = num_score
        return score

    def compute_scores_per_metric_type(
        self, 
        true_data : List[Dict], 
        pred_data : List[Dict], 
        field_metric_lookup: Dict
    ):
        """
        Compute scores per metric type, e.g. closed set, exact match.

        Args:
            true_data: True list of extracted fields
            pred_data: Predicted list of extracted fields
            field_metric_lookup: Mapping of fields to metric types.
            
        Returns:
            Dict[str, float]: mean score per metric type
        """
        scores_per_row = []
        fields = list(field_metric_lookup.keys())
        for true_data_row, pred_data_row in zip(true_data, pred_data):
            scores_per_type = defaultdict(list)
            for field in fields:
                metric_type = field_metric_lookup[field]
                true_val = true_data_row[field]
                pred_val = pred_data_row[field]
                score = self.compute_score_by_metric_type(
                    true_val, pred_val, 
                    metric_type,
                )
                scores_per_type[metric_type].append(score)
            ## compute mean score per type
            scores_per_type = {
                k : np.mean(v) for k,v in scores_per_type.items()
            }
            scores_per_row.append(scores_per_type)
        metric_types = set([field_metric_lookup[field] for field in fields])
        scores_per_type = {
            metric_type : float(np.mean([scores[metric_type] for scores in scores_per_row]))
            for metric_type in metric_types
        }
        return scores_per_type

    def compute_custom_weight_score(
        self, true_data, pred_data,
        category_data_lookup
    ):
        """
        Compute custom weighted score.

        Args:
            true_data: True list of extracted fields
            pred_data: Predicted list of extracted fields
            category_data: Mapping of categories to fields and metric types.
            
        Returns:
            Dict[str, float]: mean score per metric type
        """
        scores_per_category = defaultdict(list)
        categories = list(category_data_lookup.keys())
        for true_data_row, pred_data_row in zip(true_data, pred_data):
            for category in categories:
                for field_data in category_data_lookup[category]['fields']:
                    field = field_data['field']
                    metric_type = field_data['metric_type']
                    true_val = true_data_row[field]
                    pred_val = pred_data_row[field]
                    score = self.compute_score_by_metric_type(
                        true_val, pred_val, metric_type,
                    )
                    scores_per_category[category].append(score)
        ## get mean per category, compute weighted average
        mean_score_per_category = np.array(list(map(
            lambda x: float(np.mean(scores_per_category[x])),
            categories
        )))
        weights = np.array(list(map(
            lambda category: category_data_lookup[category]['weight'],
            categories
        )))
        ## normalize weights
        weights = weights / weights.sum()
        custom_weight_score = sum(mean_score_per_category * weights)
        return custom_weight_score


# Example usage
if __name__ == "__main__":
    # Initialize metrics evaluator
    evaluator = MetricsEvaluator()
    
    ## Example ground truth and prediction for testing
    ground_truth = "New York City"
    predicted = "NYC"
    
    # Evaluate character-level edit distance
    char_dist = evaluator.evaluate(ground_truth, predicted, "char_edit_distance")
    print(f"Character edit distance: {char_dist}")
    
    # Evaluate word-level edit distance
    word_dist = evaluator.evaluate(ground_truth, predicted, "word_edit_distance")
    print(f"Word edit distance: {word_dist}")
    
    # Example of batch evaluation
    
    # Batch evaluate precision, recall, and F1
    metrics_to_compute = [
        "precision", "recall", "f1",
        "char_edit_distance", "word_edit_distance", 
        "numerical_error","geo_distance", "time_difference" 
    ]
    results = evaluator.batch_evaluate(
        response_path="results/information_extraction/cxie_gemini/responses.json",
        metrics = metrics_to_compute,
        output_path = "results/information_extraction/cxie_gemini/scores.json"
    )
    
    # ## Aggregate metric: per metric type
    true_data = [
        {
            'Email' : 'BaseSeattlePEIS@uscg.mil',
            'Size' : '100',
            'State' : ['CA', 'California'],
            'Position' : 'Environmental Management Division',
        },
        {
            'Email' : 'BLM_HQ_GRSG_Planning@blm.gov',
            'Size' : '50',
            'State' : ['WY', 'Wyoming'],
            'Position' : 'National Sage-Grouse Conservation Coordinator',
        }
    ]
    pred_data = [
        {
            'Email' : 'BaseSeattlePEIS@uscg.mil',
            'Size' : '80',
            'State' : ['California'],
            'Position' : 'Environmental Management Division',
        },
        {
            'Email' : 'BLM_HQ_GRSG_Planning@blm.gov',
            'Size' : '45',
            'State' : ['Wisconsin'],
            'Position' : 'Conservation Coordinator',
        }
    ]
    true_data = [
        {
            'Email' : 'BaseSeattlePEIS@uscg.mil',
            'Size' : '100',
            'State' : ['CA', 'California'],
            'Position' : 'Environmental Management Division',
        },
        {
            'Email' : 'BLM_HQ_GRSG_Planning@blm.gov',
            'Size' : '50',
            'State' : ['WY', 'Wyoming'],
            'Position' : 'National Sage-Grouse Conservation Coordinator',
        }
    ]
    pred_data = [
        {
            'Email' : 'BaseSeattlePEIS@uscg.mil',
            'Size' : '80',
            'State' : ['California'],
            'Position' : 'Environmental Management Division',
        },
        {
            'Email' : 'BLM_HQ_GRSG_Planning@blm.gov',
            'Size' : '45',
            'State' : ['Wisconsin'],
            'Position' : 'Conservation Coordinator',
        }
    ]
    field_metric_lookup = {
        'Email' : 'exact_match',
        'Size' : 'numeric',
        'State' : 'closed_set',
        'Position' : 'open_set',
    }
    scores_per_type = evaluator.compute_scores_per_metric_type(
        true_data, pred_data, field_metric_lookup
    )
    ## check against ground truth scores
    true_scores = {
        'closed_set' : 0.33333,
        'exact_match' : 1.0,
        'numeric' : 0.85,
        'open_set' : 0.76666,
    }
    for metric_type, true_score in true_scores.items():
        assert isclose(scores_per_type[metric_type], scores_per_type[metric_type], rel_tol=1e-5)
    
    ## custom weight score
    category_data_lookup = {
        'important' : {
            'fields' : [
            {
                'field' : 'Email',
                'metric_type' : 'exact_match',
            },
            {
                'field' : 'Position',
                'metric_type' : 'open_set',
            },
        ],
            'weight' : 0.9,
        },
        'non_important' : {
            'fields' : [
            {
                'field' : 'Size',
                'metric_type' : 'numeric',
            },
            {
                'field' : 'State',
                'metric_type' : 'closed_set',
            },
        ],
            'weight' : 0.1,
        },
    }
    custom_weight_score = evaluator.compute_custom_weight_score(
        true_data, pred_data, category_data_lookup
    )
    assert isclose(custom_weight_score, 0.85417, rel_tol=1e-5)