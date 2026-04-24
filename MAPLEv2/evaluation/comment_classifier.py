import os
import sys
from typing import Dict, Any, Callable, Optional, List, Tuple
import time
import re

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import CommentClassificationBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_comment_classifier.log")

logger = LogManager.get_logger("comment_classifier")
    

class CommentClassifier(BaseEvaluator):
    """
    Evaluator for the Comment Classification task.
    Uses the BaseEvaluator interface and the ResponseProcessor helper.
    """

    def prepare_prompt(self, entry:CommentClassificationBenchmarkEntry, **kwargs):
        # Update the project description
        if not entry.project_description:
            if entry.project_file:
                try:
                    with open(entry.project_file, 'r', encoding='utf-8') as file:
                        project_description = file.read()
                except Exception as e:
                    logger.error(f"Error loading project description: {str(e)}")
                    raise
            else:
                logger.error(f"Missing project description as well as file with the description")
                raise
            
            entry.update_field('project_description', project_description)
            logger.info(f"Loaded project description with {len(project_description)} characters")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for comment classification (length: {len(prompt)})")
            return prompt, None
        
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
        self,
        entry: CommentClassificationBenchmarkEntry,
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
            Dictionary with comment classification results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating metadata entry with ID: {entry.entry_id}")

        # Prepare the prompt
        prompt, _ = self.prepare_prompt(entry)

        # Return None if no prompt is created
        if not prompt:
            return None
        
        answer_predicted = self.process_comment_class_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )
        if answer_predicted == None:
            logger.warning(f"No valid scope classification produced for entry_id={entry.entry_id}")
            return None
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": answer_predicted,
            "answer_expected": entry.in_scope,
        }
    
    def _clean_boolean_response(self, cleaned_response: str) -> bool:
        # Direct boolean matches
        if cleaned_response in ['true', '1', 'yes']:
            return True
        elif cleaned_response in ['false', '0', 'no']:
            return False
        
        # Check for scope-related terms
        if any(term in cleaned_response for term in ['in scope', 'inscope', 'within scope', 'relevant', 'applicable']):
            return True
        elif any(term in cleaned_response for term in ['out of scope', 'outofscope', 'outside scope', 'not scope', 'irrelevant', 'not applicable']):
            return False
        
        # Check for positive/negative indicators
        positive_terms = ['yes', 'correct', 'right', 'valid', 'appropriate', 'related', 'pertinent', 'true']
        negative_terms = ['no', 'incorrect', 'wrong', 'invalid', 'inappropriate', 'unrelated', 'not pertinent', 'false']
        
        if any(term in cleaned_response for term in positive_terms):
            return True
        elif any(term in cleaned_response for term in negative_terms):
            return False
        
        # If response contains both true and false, return None (ambiguous)
        if 'true' in cleaned_response and 'false' in cleaned_response:
            return None
        
        # For undetermined cases, return None
        return None
    
    def process_comment_class_with_retries(
        self,
        get_response_func: Callable[[], str],
        max_attempts: Optional[int] = None, 
        retry_delay: Optional[int] = None
    ) -> Optional[bool]:
        """
        Generate an LLM response that must be boolean value.

        Args:
            get_response_func: Function to get the response (no arguments).
            max_attempts (Optional[int]): Maximum number of attempts, defaults to self.max_retries.
            retry_delay (Optional[int]): Delay for retries in seconds, defaults to self.retry_delay
        
        Returns:
            bool: The accepted bin label.
            none: If no valid reponse after all atempts.
        """
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Bin attempt {attempt}/{max_attempts}")
            try:
                raw = get_response_func().strip()

                # Remove special characters and normalize
                cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', raw)
                cleaned = re.sub(r'\s+', ' ', cleaned.lower().strip())

                # Clean boolean response from LLM response
                boolean_result = self._clean_boolean_response(cleaned)
                if boolean_result is not None:
                    logger.info(f"Valid boolean response received on attempt {attempt}: {boolean_result}")
                    return boolean_result
                
                # Log the invalid response
                logger.info(f"Attempt {attempt}: Invalid response '{raw}' -> {boolean_result}")
                

            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {e}")
            
            if attempt < max_attempts:
                time.sleep(retry_delay)
        
        logger.warning(f"All {max_attempts} attempts failed. Returning fallback value: {None}")
        return None
    
    def extract_response(
            self, 
            entry: CommentClassificationBenchmarkEntry, 
            response: str
            ) -> Optional[Dict[str, Any]]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (MapClassifyBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Optional[Dict[str, Any]]): Dictionary of processed result if valid, otherwise None
        """

        # Remove special characters and normalize
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', response)
        cleaned = re.sub(r'\s+', ' ', cleaned.lower().strip())

        # Clean boolean response from LLM response
        boolean_result = self._clean_boolean_response(cleaned)
        if boolean_result is not None:
            logger.info(f"Accepted boolean response: {boolean_result}")
            
            # Create result dictionary
            result = {
                "entry_id": entry.entry_id,
                "answer_predicted": boolean_result,
                "answer_expected": entry.in_scope,
            }
            return result

        else:
            logger.warning(f"Rejected response '{response}', cannot extract boolean response")
            return None
                    
            