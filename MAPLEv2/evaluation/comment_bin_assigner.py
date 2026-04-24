import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Callable
import json
import time
from difflib import get_close_matches

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import CommentBinBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_bin_assigner.log")

logger = LogManager.get_logger("bin_assigner")

def extract_bins_with_descriptions(json_file_path: str) -> Tuple[str,List[str]]:
    """
    Extract bin names and descriptions from a JSON file and format them as a string.
    
    Args:
        json_file_path (str): Path to the JSON file containing bin names as keys 
                             and descriptions as values.
    
    Returns:
        str: Formatted string with bin names and descriptions ready for prompt insertion.
        List[str]: List of bin names
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            bins_data = json.load(file)
        
        if not isinstance(bins_data, dict):
            logger.error("JSON file must contain a dictionary with bin names as keys")
            raise ValueError("JSON file must contain a dictionary with bin names as keys")
        
        # Format bins with descriptions
        formatted_bins = []
        formatted_bin_names = ["Ensure that the bin name is ONE of the following"]
        for bin_name, description in bins_data.items():
            if description != None:
                formatted_bins.append(f"- {bin_name}: {description}")
                formatted_bin_names.append(f"- {bin_name}")
            else:
                formatted_bins.append(f"- {bin_name}")
                formatted_bin_names.append(f"- {bin_name}")
        
        formatted_bins_block = "\n".join(formatted_bins) + "\n\n" + "\n".join(formatted_bin_names)
        accepted_bins = list(bins_data.keys())
        return formatted_bins_block, accepted_bins
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {json_file_path} - {str(e)}")
        raise json.JSONDecodeError(f"Invalid JSON file: {json_file_path} - {str(e)}")
    except Exception as e:
        logger.error(f"Error processing bins file: {str(e)}")
        raise Exception(f"Error processing bins file: {str(e)}")
    

class CommentBinAssigner(BaseEvaluator):
    """
    Evaluator for the Comment Bin Assignment task.
    Uses the BaseEvaluator interface and the ResponseProcessor helper.
    """

    def prepare_prompt(self, entry:CommentBinBenchmarkEntry,**kwargs) -> Tuple[str,Optional[List[str]]]:
        # Update the bins_block
        if not entry.bins_guidance_block or not entry.all_bins:
            bins_guidance_block, all_bins = extract_bins_with_descriptions(entry.binning_guidance_file)
            entry.update_field('bins_guidance_block', bins_guidance_block)
            entry.update_field('all_bins', all_bins)
            logger.info(f"Loaded binning guidance description with {len(bins_guidance_block)} characters")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for comment bin assignment (length: {len(prompt)})")
            return prompt,None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
        self,
        entry: CommentBinBenchmarkEntry,
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
        
        # Format the prompt
        prompt,_ = self.prepare_prompt(entry)
        
        # Return None if no prompt is created
        if not prompt:
            return None

        predicted_bin = self.process_bin_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            allowed_bins=entry.all_bins,
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )

        if predicted_bin is None:
            logger.warning(f"No valid bin produced for entry_id={entry.entry_id}")
            return None
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": predicted_bin,
            "answer_expected": entry.bin,
        }
    
    def process_bin_response_with_retries(
        self,
        get_response_func: Callable[[], str],
        allowed_bins: List[str],
        max_attempts: Optional[int] = None, 
        retry_delay: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate an LLM response that must be exactly one of the permitted bin names.

        Args:
            get_response_func: Function to get the response (no arguments).
            allowed_bins: List of valid bin labels (case-sensitive).
            max_attempts (Optional[int]): Maximum number of attempts, defaults to self.max_retries.
            retry_delay (Optional[int]): Delay for retries in seconds, defaults to self.retry_delay
        
        Returns:
            str: The accepted bin label.
            none: If no valid reponse after all atempts.
        """
        subcategory_bins = {
            bin_name.split(':')[-1].strip(): bin_name \
            for bin_name in allowed_bins
        }

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Bin attempt {attempt}/{max_attempts}")
            try:
                candidate = None
                raw = get_response_func().strip()

                # Only keep first line to avoid explanations
                for line in raw.splitlines():
                    candidate = line.strip() 
                    
                    if candidate in allowed_bins:
                        logger.info(f"Accepted bin response: {candidate}")
                        return candidate
                    
                    # Second, check if input matches the part after the colon
                    elif candidate in subcategory_bins.keys():
                        clean_bin_name = subcategory_bins[candidate]
                        logger.info(f"Accepted bin response: {candidate} as {clean_bin_name}")
                        return clean_bin_name
                    
                    else:
                        matches = get_close_matches(candidate, allowed_bins, n=1, cutoff=0.6)
                        if not matches:
                            logger.warning(
                                f"Rejected bin '{candidate}'. Obtained from {candidate[:50]}"
                            )
                        else:
                            matched_name = matches[0]
                            logger.info(f"Accepted bin response: {candidate} as {matched_name}")
                            return matched_name

            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {e}")
            
            if attempt < max_attempts:
                time.sleep(retry_delay)
        
        # Fallback value
        if candidate:
            logger.warning(
                f"No valid bin after {max_attempts} attempts - returning latest identified bin {candidate}"
            )
            return candidate
        else:
            return None
        
    def extract_response(
            self, 
            entry: CommentBinBenchmarkEntry, 
            response: str
            ) -> Optional[Dict[str, Any]]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (CommentBinBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Optional[Dict[str, Any]]): Dictionary of processed result if valid, otherwise None
        """      
        # Validate and extract class name
        matches = get_close_matches(response, entry.all_bins, n=1, cutoff=0.6)
        if not matches:
            logger.warning(f"Rejected bin '{response}'")
            return None
        else:
            matched_name = matches[0]
            logger.info(f"Accepted bin response: {response} as {matched_name}")
            
            # Create result dictionary
            result = {
                "entry_id": entry.entry_id,
                "answer_predicted": matched_name,
                "answer_expected": entry.bin,
            }
            return result