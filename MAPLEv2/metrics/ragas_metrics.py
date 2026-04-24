import os
import sys
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import math

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_similarity,
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)
from tenacity import retry, stop_after_attempt, wait_exponential

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers.base_handler import BaseLLMHandler
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_metric.log")

logger = LogManager.get_logger(__name__)

class RAGAS_Evaluator:
    """
    A class for evaluating language model responses using RAGAS metrics.

    This class processes results from language model responses and evaluates them
    using various RAGAS metrics such as answer similarity, correctness, and context precision/recall.
    """
    # Define available RAGAS metrics
    # NOTE: Add additional metrics for latest version of RAGAS

    def __init__(
            self, 
            llm_handler: BaseLLMHandler, 
            embedding_model_name: str,
            ):
        """
        Initialize the RAGAS_Evaluator.

        Args:
            llm_handler (BaseLLMHandler): Handler for the language model.
            embedding_model_name (str): Name of the embedding model to use.
            metrics (List[str], optional): List of metrics to evaluate. Defaults to all available metrics.
        
        Raises:
            ValueError: If any specified metric is not available.
        """
        self.llm_handler = llm_handler
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Available RAGAS metrics mapping
        self.available_metrics = {
            'answer_correctness': answer_correctness,
            'answer_similarity': answer_similarity,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'faithfulness': faithfulness
        }
        
    
    def load_responses_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load responses from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing responses
            
        Returns:
            List[Dict[str, Any]]: List of response entries
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file format is invalid
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Response JSON file not found: {json_path}")
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                response_dict = json.load(f)
                
            # Convert dictionary to list of entries
            responses = []
            for entry_id, entry_data in response_dict.items():
                # Add the entry_id to the entry data for reference
                entry_data['entry_id'] = entry_id
                responses.append(entry_data)
                
            logger.info(f"Loaded {len(responses)} responses from {json_path}")
            return responses
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {json_path} - {str(e)}")
            raise ValueError(f"Invalid JSON file format: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading responses: {str(e)}")
            raise
    
    def load_existing_scores(self, output_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load existing RAGAS scores from output file.
        
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
            return scores_data
            
        except Exception as e:
            logger.warning(f"Error loading existing scores: {str(e)}. Starting fresh evaluation.")
            return {}
        
    def prepare_ragas_dataset(self, responses: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare a RAGAS dataset from response entries.
        
        Args:
            responses: List of response entries
            
        Returns:
            Dataset: RAGAS-compatible dataset
        """
        # Prepare data for RAGAS evaluation
        questions = []
        answers = []
        ground_truths = []
        contexts = []
        
        for response in responses:
            questions.append(response.get('question', ''))
            answers.append(response.get('answer_predicted', ''))
            ground_truths.append(response.get('answer_expected', ''))
            
            # Combine chunks into context (filter out empty chunks)
            chunks = response.get('chunks', ['', '', ''])
            context_list = [chunk for chunk in chunks if chunk and chunk.strip()]
            contexts.append(context_list if context_list else [''])
        
        # Create RAGAS dataset
        '''
        dataset_dict = {
            'question': questions,
            'answer': answers,
            'ground_truth': ground_truths,
            'contexts': contexts
        }
        '''

        dataset_dict = {
           'question': [str(q) for q in questions],
           'answer': [str(a) for a in answers], 
           'ground_truth': [str(gt) for gt in ground_truths],
           'contexts': [[str(ctx) for ctx in context_list] for context_list in contexts]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Prepared RAGAS dataset with {len(dataset)} entries")
        return dataset
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=15, max=60))
    def evaluate_with_retry(self, metrics:List[str], responses:List[Dict[str,Any]]):
        """
        Evaluate the dataset using RAGAS metrics with retry logic.

        Args:
            metrics (List[str]): List of RAGAS metrics to use for evaluation.
            responses (List[Dict[str,Any]]): The list of responses to evaluate.

        Returns:
            The evaluation result.

        Raises:
            Exception: If evaluation fails after all retry attempts.
        """
        # Validate metrics
        for metric in metrics:
            if metric not in self.available_metrics:
                available = list(self.available_metrics.keys())
                raise ValueError(f"Metric '{metric}' not available. Available metrics: {available}")
        
        # Prepare dataset
        dataset = self.prepare_ragas_dataset(responses)
        
        # Get metric objects
        metric_objects = [self.available_metrics[metric] for metric in metrics]
        
        logger.info(f"Computing RAGAS scores for {len(responses)} entries with metrics: {metrics}")
        try:
            result = evaluate(
                dataset,metric_objects,
                llm=self.llm_handler,
                embeddings=self.embedding_model
            )
            # Extract scores
            scores_df = pd.DataFrame(result.scores)
            scores = {}
            for metric in metrics:
                if metric in scores_df.columns:
                    scores[metric] = scores_df[metric]
                else:
                    logger.warning(f"Metric '{metric}' not found in RAGAS results")
                    scores[metric] = [0.0] * len(responses)
            
            logger.info("RAGAS evaluation completed successfully")
            return scores
        
        except Exception as e:
            logger.warning(f"Error during evaluation: {str(e)}")
            raise  # This will trigger a retry

    def _is_nan_value(self, value):
        """Check if value is any type of NaN"""
        try:
            return (
                (isinstance(value, float) and math.isnan(value)) or
                pd.isna(value) or
                (hasattr(value, '__name__') and value.__name__ == 'nan')
            )
        except (TypeError, AttributeError):
            return False

    def evaluate_responses(
        self,
        response_json_path: str,
        output_scores_path: str,
        metrics: List[str],
        batch_size: int = 4,
        continue_from_previous: bool = True
    ) -> str:
        """
        Evaluate responses with RAGAS metrics and save scores.
        
        Args:
            response_json_path: Path to the response JSON file
            output_scores_path: Path to save the computed scores
            metrics: List of RAGAS metrics to compute
            batch_size: Batch size for RAGAS evaluation
            continue_from_previous: Whether to continue from previous evaluation
            
        Returns:
            str: Path to the saved scores file
            
        Raises:
            FileNotFoundError: If response JSON file doesn't exist
            ValueError: If metrics are invalid
        """
        logger.info(f"Starting RAGAS evaluation for responses from: {response_json_path}")
        logger.info(f"Metrics to compute: {metrics}")
        logger.info(f"Output path: {output_scores_path}")
        logger.info(f"Continue from previous: {continue_from_previous}")
        
        # Load responses
        all_responses = self.load_responses_from_json(response_json_path)
        
        # Load existing scores if continuing from previous
        existing_scores = {}
        responses_to_evaluate = all_responses
        
        if continue_from_previous:
            existing_scores = self.load_existing_scores(output_scores_path)
            
            # Filter out responses that already have scores
            responses_to_evaluate = []
            for response in all_responses:
                entry_id = response['entry_id']
                if entry_id not in existing_scores:
                    responses_to_evaluate.append(response)
                else:
                    logger.debug(f"Skipping entry {entry_id} - already evaluated")
            
            logger.info(f"Found {len(existing_scores)} existing scores")
            logger.info(f"Need to evaluate {len(responses_to_evaluate)} remaining responses")
        
        # Compute scores for remaining responses
        if responses_to_evaluate:
            logger.info(f"Computing RAGAS scores for {len(responses_to_evaluate)} responses")
            
            # Ensure output directory exists before processing batches
            output_dir = os.path.dirname(output_scores_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Process in batches
            remaining_responses = responses_to_evaluate.copy()
            processed_count = 0
            
            while remaining_responses:
                # Take a batch
                current_batch = remaining_responses[:batch_size]
                remaining_responses = remaining_responses[batch_size:]
                
                batch_num = (processed_count // batch_size) + 1
                total_batches = (len(responses_to_evaluate) + batch_size - 1) // batch_size
                logger.info(f"Processing batch {batch_num}/{total_batches} with {len(current_batch)} responses")
                
                try:
                    # Compute scores for this batch
                    batch_scores = self.evaluate_with_retry(metrics, current_batch)
                    
                    # Create individual entry scores
                    for i, response in enumerate(current_batch):
                        entry_id = response['entry_id']
                        result = response.copy()
                        result["scores"] = {}
                        
                        # Add computed scores
                        for metric in metrics:
                            if self._is_nan_value(batch_scores[metric][i]):
                                result["scores"][metric] = None
                            else:
                                result["scores"][metric] = batch_scores[metric][i]
                        
                        existing_scores[entry_id] = result
                    
                    # Save progress after each batch
                    with open(output_scores_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_scores, f, indent=2)
                    
                    processed_count += len(current_batch)
                    logger.info(f"Batch {batch_num}/{total_batches} completed. Progress saved to {output_scores_path}")
                    logger.info(f"Total processed: {processed_count}/{len(responses_to_evaluate)} responses")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    logger.info(f"Saving progress up to batch {batch_num-1}")
                    
                    # Save current progress even if this batch failed
                    with open(output_scores_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_scores, f, indent=2)
                    
                    # Re-raise the exception to stop processing
                    raise
        
        else:
            logger.info("No new responses to evaluate - all entries already have scores")
        
        # Final save (in case no new responses were processed)
        if not responses_to_evaluate:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_scores_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save existing scores
            with open(output_scores_path, 'w', encoding='utf-8') as f:
                json.dump(existing_scores, f, indent=2)
        
        logger.info(f"Final save: RAGAS scores for {len(existing_scores)} total entries saved to {output_scores_path}")
        
        return output_scores_path
