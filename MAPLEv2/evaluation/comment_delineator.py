import os
import sys
from typing import Dict, Any, List, Optional, Callable
import json
import time

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import CommentDelineateBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager
from utils import validate_quotes_in_text, validate_extracted_information

if __name__ == "__main__":
    LogManager.initialize("logs/test_comment_delineator.log")

logger = LogManager.get_logger("comment_delineator")


class CommentDelineator(BaseEvaluator):
    """
    Evaluator for the Comment Delineation task.
    Uses the BaseEvaluator interface.
    """

    def prepare_prompt(self, entry:CommentDelineateBenchmarkEntry, **kwargs):
        # Update the bins_block
        if not entry.full_comment:
            try:
                with open(entry.comment_file, 'r', encoding='utf-8') as file:
                    full_comment = file.read()
            except FileNotFoundError:
                logger.error(f"Text file not found: {entry.comment_file}")
                raise FileNotFoundError(f"Text file not found: {entry.comment_file}")
            except Exception as e:
                logger.error(f"Error processing comment file: {str(e)}")
                raise Exception(f"Error processing comment file: {str(e)}")
            
            entry.update_field('full_comment', full_comment)
            logger.info(f"Loaded full comment with {len(full_comment)} characters")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for comment bin assignment (length: {len(prompt)})")
            return prompt, None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
        self,
        entry: CommentDelineateBenchmarkEntry,
        **kwargs
        ) -> Dict[str, Any]:
        """
        Run the LLM on one entry and return the chosen bin.
        Retries are handled by ResponseProcessor.process_bin_response_with_retries().

        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            max_attempts: Max attempts to generate a valid response.
            
        Returns:
            Dictionary with comment bin assignment results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating metadata entry with ID: {entry.entry_id}")
        
        # Prepare the prompt
        prompt,_ = self.prepare_prompt(entry)

        # Return None if prompt is None
        if not prompt:
            return None

        predicted_quotes = self.process_quote_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            entire_text=entry.full_comment,
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )

        if not predicted_quotes:
            logger.warning(f"No quotes identified for entry_id={entry.entry_id}")
            return None
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": predicted_quotes,
            "answer_expected": entry.comment_quotes,
        }
    
    def process_quote_response_with_retries(
        self,
        get_response_func: Callable[[], str],
        entire_text: str,
        max_attempts: Optional[int] = None, 
        retry_delay: Optional[int] = None
    ) -> Optional[List[str]]:
        """
        Generate an LLM response that must be in the entire input text.

        Args:
            get_response_func: Function to get the response (no arguments).
            entire_text: Entire text from which quote needs to be extracted.
            max_attempts (Optional[int]): Maximum number of attempts, defaults to self.max_retries.
            retry_delay (Optional[int]): Delay for retries in seconds, defaults to self.retry_delay
        
        Returns:
            Optional[List[str]]: The accepted quote.
        """
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Delineation attempt {attempt}/{max_attempts}")
            try:
                raw = get_response_func().strip()

                try:
                    is_valid, quotes = validate_extracted_information(raw, field_type="array_string")
                    if is_valid:
                        validation_result = validate_quotes_in_text(quotes, entire_text)
                        return validation_result['valid_quotes']
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Unable to load response as JSON object: {e}")

            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {e}")
            
            if attempt < max_attempts:
                time.sleep(retry_delay)
        
        return None
        
    def extract_response(
            self, 
            entry: CommentDelineateBenchmarkEntry, 
            response: str
            ) -> Dict[str, Any]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (CommentDelineateBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Dict[str, Any]): Dictionary of processed result
        """
        try:
            is_valid, quotes = validate_extracted_information(response, field_type="array_string")
            if is_valid:
                validation_result = validate_quotes_in_text(quotes, entry.full_comment)
                return {
                    "entry_id": entry.entry_id,
                    "answer_predicted": validation_result['valid_quotes'],
                    "answer_expected": entry.comment_quotes,
                } 
            
        except json.JSONDecodeError as e:
            logger.error(f"Unable to load response as JSON object: {e}")
            return None