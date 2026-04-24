import os
import sys
from typing import Dict, Any, List, Tuple, Callable, Optional
import json
import time
from difflib import get_close_matches

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import MapClassifyBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_map_classifier.log")

logger = LogManager.get_logger("map_classifier")

def extract_classification_description(classes: List[str]) -> str:
    """
    Extract class names and format it as a string to be used in a prompt.
    
    Args:
        classes (List[str]): List of classes.
    
    Returns:
        str: Formatted string with class names and descriptions ready for prompt insertion.
        
    """
    try:
        if not isinstance(classes, list):
            logger.error("'classes' should be a list of strings which are the classes")
            raise ValueError("'classes' should be a list of strings which are the classes")
        
        return "\n".join(f"- {class_name}" for class_name in classes)
        
    except Exception as e:
        logger.error(f"Error processing classes: {str(e)}")
        raise Exception(f"Error processing classes: {str(e)}")

def extract_classes_with_descriptions(json_file_path: str) -> Tuple[str,List[str]]:
    """
    Extract class names and descriptions from a JSON file and format them as a string.
    
    Args:
        json_file_path (str): Path to the JSON file containing class names as keys 
                             and descriptions as values.
    
    Returns:
        str: Formatted string with class names and descriptions ready for prompt insertion.
        List[str]: List of class names
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            classes_data = json.load(file)
        
        if not isinstance(classes_data, dict):
            logger.error("JSON file must contain a dictionary with class names as keys")
            raise ValueError("JSON file must contain a dictionary with class names as keys")
        
        # Format classes with descriptions
        formatted_classes = []
        for class_name, description in classes_data.items():
            if description != None:
                formatted_classes.append(f"- {class_name}: {description}")
            else:
                formatted_classes.append(f"- {class_name}")
        
        return "\n".join(formatted_classes), list(classes_data.keys())
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {json_file_path} - {str(e)}")
        raise json.JSONDecodeError(f"Invalid JSON file: {json_file_path} - {str(e)}")
    except Exception as e:
        logger.error(f"Error processing classes file: {str(e)}")
        raise Exception(f"Error processing classes file: {str(e)}")
    

class MapClassifier(BaseEvaluator):
    """
    Evaluator for the Map Classification task.
    Uses the BaseEvaluator interface and the ResponseProcessor helper.
    """

    def prepare_prompt(
            self, 
            entry:MapClassifyBenchmarkEntry,
            **kwargs
            ) -> Tuple[str,Optional[List[str]]]:

        # Update the classification guidance block
        if not entry.classification_guidance_block:
            if entry.classification_guidance_file:
                logger.info("Loading map classification guidance from specified JSON file")
                classification_guidance_block, all_classes = extract_classes_with_descriptions(entry.classification_guidance_file)
                entry.update_field('classification_guidance_block', classification_guidance_block)
                entry.update_field('all_classes', all_classes)
                logger.info(f"Loaded classification guidance description with {len(classification_guidance_block)} characters")
            elif entry.all_classes:
                logger.info("Loading map classification guidance from list of classes provided. No additional JSON file found")
                classification_guidance_block = extract_classification_description(entry.all_classes)
                entry.update_field('classification_guidance_block', classification_guidance_block)
                logger.info(f"Loaded classification guidance description with {len(classification_guidance_block)} characters")
            else:
                logger.error("No guidance provided to classify the maps")
                raise

        # Check if path to the image exists
        image_directory = kwargs.get('image_directory', None)
        if os.path.exists(entry.image_file):
            image_path = entry.image_file
        elif image_directory and entry.project and entry.pdf_name:
            image_path = f"{image_directory}{entry.project}/{entry.pdf_name}/{entry.image_file}"
            if os.path.exists(image_path):
                logger.info(f"Constructed image path: {image_path}")
            else:
                logger.error("Constructed image path does not exist.")
                return None, None
        else:
            logger.error("Invalid image file path, or the path to the image file cannot be constructed. Requires project and pdf names.")
            raise FileNotFoundError(f"Path to image file not found")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for map classification (length: {len(prompt)})")
            return prompt, [image_path]
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None
        
    def evaluate_entry(
        self,
        entry: MapClassifyBenchmarkEntry,
        **kwargs
        ) -> Dict[str, Any]:
        """
        Run the LLM on one entry and return the chosen class.
        Retries are handled by ResponseProcessor.process_bin_response_with_retries().

        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            max_attempts: Max attempts to generate a valid response.
            
        Returns:
            Dictionary with map class assignment results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating metadata entry with ID: {entry.entry_id}")
        
        # Prepare prompt from the benchmark entry
        prompt,image_path = self.prepare_prompt(entry,**kwargs)

        # Return None if no prompt is created or no image path is identified
        if not prompt or not image_path:
            return None

        predicted_class = self.process_bin_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt, images=image_path),
            allowed_bins=entry.all_classes,
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )

        if predicted_class is None:
            logger.warning(f"No valid class produced for entry_id={entry.entry_id}")
            return None
        
        return {
            "entry_id": entry.entry_id,
            "answer_predicted": predicted_class,
            "answer_expected": entry.correct_class,
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
            entry: MapClassifyBenchmarkEntry, 
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
        # Validate and extract class name
        matches = get_close_matches(response, entry.all_classes, n=1, cutoff=0.6)
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
                "answer_expected": entry.correct_class,
            }
            return result