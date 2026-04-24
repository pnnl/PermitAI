import os
import sys
from typing import Dict, Any, Optional, Callable, List
import time
import re

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import BinSummarizerBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_bin_summarizer.log")

logger = LogManager.get_logger("bin_summarizer")

class BinSummarizer(BaseEvaluator):
    """
    Evaluator for the Comment Bin Summarization task.
    Uses the BaseEvaluator interface and the ResponseProcessor helper.
    """

    def prepare_prompt(self, entry:BinSummarizerBenchmarkEntry, **kwargs):
        # Update the bins_block
        if not entry.comments_block:
            comments_block = '\n'.join(f'- {comment}' for comment in entry.comments)
            entry.update_field('comments_block', comments_block)
            logger.info(f"Loaded all the comments with {len(comments_block)} characters")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for bin summarization (length: {len(prompt)})")
            return prompt, None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
        self,
        entry: BinSummarizerBenchmarkEntry,
        **kwargs
        ) -> Dict[str, Any]:
        """
        Run the LLM on one entry and return the chosen class for the comment.
        Retries are handled by ResponseProcessor.process_comment_class_with_retries().

        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            max_attempts: Max attempts to generate a valid response.
            retry_delay: Duration in seconds between retries
            
        Returns:
            Dictionary with comment bin summarization results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating metadata entry with ID: {entry.entry_id}")

        # Prepare the prompt
        prompt, _ = self.prepare_prompt(entry, **kwargs)
        
        # Return None if no prompt is created
        if not prompt:
            return None

        answer_predicted = self.process_bin_summary_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": answer_predicted,
            "answer_expected": entry.summary,
        }
    
    def process_bin_summary_with_retries(
        self,
        get_response_func: Callable[[], str],
        max_attempts: Optional[int] = None, 
        retry_delay: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate an LLM response that must be exactly one of the permitted bin names.

        Args:
            get_response_func: Function to get the response (no arguments).
            max_attempts (Optional[int]): Maximum number of attempts
            retry_delay (Optional[int]): Delay for retries in seconds
        
        Returns:
            str: The accepted bin label.
            none: If no valid reponse after all atempts.
        """
        for attempt in range(1, max_attempts+1):
            try:
                logger.info(f"Attempt {attempt}/{max_attempts} to generate response")
                raw = get_response_func().strip()
                
                # Remove special characters and normalize
                cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', raw)
                
                # Remove HTML-like decorators
                cleaned = re.sub(r'<<[^>]*>>', '', cleaned)  # Remove opening tags
                cleaned = re.sub(r'<</[^>]*>>', '', cleaned)  # Remove closing tags
                
                # convert to lower case and strip 
                cleaned = re.sub(r'\s+', ' ', cleaned.lower().strip())
                return cleaned
                
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {str(e)}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
        
        logger.warning(f"Failed to generate valid response after {max_attempts} attempts")
        return ""
    
    def extract_response(self, 
            entry: BinSummarizerBenchmarkEntry, 
            response: str
            ) -> Dict[str, Any]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (BinSummarizerBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Dict[str, Any]): Dictionary of processed result
        """
        # Remove special characters and normalize
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', response)
        
        # Remove HTML-like decorators
        cleaned = re.sub(r'<<[^>]*>>', '', cleaned)  # Remove opening tags
        cleaned = re.sub(r'<</[^>]*>>', '', cleaned)  # Remove closing tags
        
        # convert to lower case and strip 
        cleaned = re.sub(r'\s+', ' ', cleaned.lower().strip())
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": cleaned,
            "answer_expected": entry.summary,
        }