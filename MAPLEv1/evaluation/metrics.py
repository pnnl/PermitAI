import os
import sys
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

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

from utils import load_results_csv
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
    AVAILABLE_METRICS = {
        'answer_similarity': answer_similarity,
        'answer_correctness': answer_correctness,
        'context_precision': context_precision,
        'context_recall': context_recall,
        'faithfulness': faithfulness
    }

    def __init__(
            self, results_csv_path: str, 
            llm_handler: BaseLLMHandler, 
            embedding_model_name: str,
            metrics: List[str] = None):
        """
        Initialize the RAGAS_Evaluator.

        Args:
            results_csv_path (str): Path to the CSV file containing the results.
            llm_handler (BaseLLMHandler): Handler for the language model.
            embedding_model_name (str): Name of the embedding model to use.
            metrics (List[str], optional): List of metrics to evaluate. Defaults to all available metrics.
        
        Raises:
            ValueError: If any specified metric is not available.
        """
        self.dataset = load_results_csv(results_csv_path)
        self.llm_handler = llm_handler
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize metrics list
        if metrics is None:
            self.metrics = list(self.AVAILABLE_METRICS.values())
            self.metric_names = list(self.AVAILABLE_METRICS.keys())
        else:
            self.metrics = []
            self.metric_names = []
            for metric in metrics:
                if metric not in self.AVAILABLE_METRICS:
                    raise ValueError(f"Metric '{metric}' is not available. Available metrics: {list(self.AVAILABLE_METRICS.keys())}")
                self.metrics.append(self.AVAILABLE_METRICS[metric])
                self.metric_names.append(metric)
        logger.info(f"Metrics to be evaluated: {self.metric_names}")
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=15, max=60))
    def evaluate_with_retry(self, metrics, dataset):
        """
        Evaluate the dataset using RAGAS metrics with retry logic.

        Args:
            metrics (list): List of RAGAS metrics to use for evaluation.
            dataset (Dataset): The dataset to evaluate.

        Returns:
            The evaluation result.

        Raises:
            Exception: If evaluation fails after all retry attempts.
        """
        try:
            result = evaluate(
                dataset,metrics,
                llm=self.llm_handler,
                embeddings=self.embedding_model
            )
            return result
        except Exception as e:
            logger.warning(f"Error during evaluation: {str(e)}")
            raise  # This will trigger a retry

    def evaluate_results(self, output_csv_path: str, batch_size: int = 20, continue_from_previous: bool = True):
        """
        Evaluate the results and save the evaluation scores to a CSV file.

        This method processes the dataset in batches, evaluates each batch using RAGAS metrics,
        and appends the results to the output CSV file.

        Args:
            output_csv_path (str): Path to save the output CSV file with evaluation scores.
            batch_size (int, optional): Number of items to process in each batch. Defaults to 20.
            continue_from_previous (bool, Optional): Start metric computation from last checkpoint. Defaults to true.

        Raises:
            Exception: If evaluation fails for a batch after all retry attempts.
        """

        # Define standard column order
        standard_columns = [
            'file_name', 'page_number', 'question_type', 'question',
            'answer_expected', 'answer_predicted'
        ]
        # Add selected metrics to columns
        standard_columns.extend(self.metric_names)

        # Determine the starting index
        if continue_from_previous:
            if not os.path.exists(output_csv_path):
                logger.info(f"No file {output_csv_path} found with previous scores")
                start_index = 0
            else:
                existing_df = pd.read_csv(output_csv_path)
                start_index = len(existing_df)
                logger.info(f"Existing output score file found. Starting from index {start_index}")
        else:
            logger.info("Starting metric evaluation for all questions.")
            start_index = 0

        # Process in batches
        for batch_start in range(start_index, len(self.dataset.questions), batch_size):
            batch_end = min(batch_start + batch_size, len(self.dataset.questions))
            logger.info(f"Processing batch from index {batch_start} to {batch_end}")

            # Evaluate using RAGAS
            batch_dataset = Dataset.from_dict({
                'question' : self.dataset.questions[batch_start:batch_end], 
                'answer' : self.dataset.answers[batch_start:batch_end],
                'ground_truth' : self.dataset.ground_truths[batch_start:batch_end],
                'retrieved_contexts' : self.dataset.contexts[batch_start:batch_end]
            })
        
            try:
                logger.info(f"Starting RAGAS evaluation for batch {batch_start//batch_size + 1}")
                result = self.evaluate_with_retry(self.metrics, batch_dataset)
                logger.info(f"RAGAS evaluation completed successfully for batch {batch_start//batch_size + 1}")

                # Extract scores from the result
                scores = result.scores

                # Prepare data for output CSV
                output_data = []
                for idx in range(batch_end - batch_start):
                    entry = {}
                    
                    # Add standard fields if they exist and are not None
                    if self.dataset.files is not None:
                        entry['file_name'] = self.dataset.files[batch_start + idx]
                    if self.dataset.pages is not None:
                        entry['page_number'] = self.dataset.pages[batch_start + idx]
                    if self.dataset.qtypes is not None:
                        entry['question_type'] = self.dataset.qtypes[batch_start + idx]
                    
                    # Add required fields
                    entry.update({
                        'question': self.dataset.questions[batch_start + idx],
                        'answer_expected': self.dataset.ground_truths[batch_start + idx],
                        'answer_predicted': self.dataset.answers[batch_start + idx]
                    })
                    
                    # Add metric scores
                    for metric_name in self.metric_names:
                        if metric_name in scores.features:
                            entry[metric_name] = scores[metric_name][idx]

                    # Only add non-None values
                    output_data.append({k: v for k, v in entry.items() if v is not None})

                # Create and save output DataFrame
                output_df = pd.DataFrame(output_data)
                
                # Reorder columns (only include columns that exist)
                existing_columns = [col for col in standard_columns if col in output_df.columns]
                output_df = output_df[existing_columns]

                # Save output DataFrame
                if batch_start == 0:
                    output_df.to_csv(
                        output_csv_path, 
                        mode='w', 
                        header=True, 
                        index=False
                        )
                    logger.info(f"Evaluation results for batch {batch_start//batch_size + 1} written to {output_csv_path}")
                else:
                    output_df.to_csv(
                        output_csv_path, 
                        mode='a', 
                        header=not os.path.exists(output_csv_path), 
                        index=False
                        )
                    logger.info(f"Evaluation results for batch {batch_start//batch_size + 1} appended to {output_csv_path}")

            except Exception as e:
                logger.error(f"Failed to complete RAGAS evaluation for batch {batch_start//batch_size + 1}: {str(e)}")
                raise
    
    def reevaluate_results(self, scores_csv: str, column_names: list = ["answer_correctness"], batch_size: int = 20):
        """
        Re-evaluate rows with NaN values in specified columns and update the scores CSV.

        This method checks for rows with NaN values in the specified columns, 
        re-evaluates those entries, and updates the scores CSV with the new results.

        Args:
            scores_csv (str): Path to the CSV file containing the evaluation scores.
            column_names (list): List of column names to check for NaN values.

        Raises:
            FileNotFoundError: If the scores CSV file is not found.
            KeyError: If any of the specified column names are not in the CSV file.
        """
        # Define metrics
        metrics = []
        if "answer_similarity" in column_names:
            metrics.append(answer_similarity)
        if "answer_correctness" in column_names:
            metrics.append(answer_correctness)
        if "context_precision" in column_names:
            metrics.append(context_precision)
        if "context_recall" in column_names:
            metrics.append(context_recall)

        logger.info(f"Reevaluating for metrics: {[m.name for m in metrics]}")
        try:
            # Read the scores CSV
            df = pd.read_csv(scores_csv)
            
            # Check for NaN values in specified columns
            nan_rows = df[df[column_names].isna().any(axis=1)]
            nan_indices = nan_rows.index.tolist()
            
            if not nan_indices:
                logger.info("No rows with NaN values found in the specified columns.")
                return

            logger.info(f"Found {len(nan_indices)} rows with NaN values to re-evaluate.")

            # Process in batches
            for i in range(0, len(nan_indices), batch_size):
                batch_indices = nan_indices[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {len(nan_indices)//batch_size + 1}")

                # Extract corresponding entries from the dataset for this batch
                questions = [self.dataset.questions[j] for j in batch_indices]
                answers = [self.dataset.answers[j] for j in batch_indices]
                ground_truths = [self.dataset.ground_truths[j] for j in batch_indices]
                contexts = [self.dataset.contexts[j] for j in batch_indices]

                # Prepare the dataset for re-evaluation
                reeval_dataset = Dataset.from_dict({
                    'question': questions,
                    'answer': answers,
                    'ground_truth': ground_truths,
                    'retrieved_contexts': contexts
                })


                # Re-evaluate
                result = self.evaluate_with_retry(metrics, reeval_dataset)
                scores = result.scores

                # Update the DataFrame with new scores
                for col in column_names:
                    if col in scores.features:
                        df.loc[batch_indices, col] = scores[col]

                # Save the updated DataFrame
                df.to_csv(scores_csv, index=False)
                logger.info(f"Updated scores saved to {scores_csv}")

        except FileNotFoundError:
            logger.error(f"Scores CSV file not found: {scores_csv}")
            raise
        except KeyError as e:
            logger.error(f"Column not found in CSV: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during re-evaluation: {str(e)}")
            raise
