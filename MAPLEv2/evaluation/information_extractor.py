import os
import sys
from typing import Dict, Any, List, Callable, Optional, Tuple
import time

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import IEBenchmarkEntry
from utils import get_pdf_context_from_json
from utils import validate_extracted_information
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_information_evaluator.log")

logger = LogManager.get_logger("information_extractor")


class InformationExtractor(BaseEvaluator):
    """
    Evaluator for information-extraction benchmarks.
    """

    def prepare_prompt(
            self,
            entry: IEBenchmarkEntry,
            **kwargs
        ) -> Tuple[str,Optional[List[str]]]:
        """
        Prepare prompt for the benchmark entry

        Args:
            entry (IEBenchmarkEntry): The benchmark entry to evaluate.
        kwargs:
            json_directory (str): Directory with JSON chunks.
            
        Returns:
            prompt (str): Formatted prompt with document text.
        """
        # Keyword arguments
        json_directory = kwargs.get('json_directory', None)

        # Check if entry has attribute 'document_text'
        if not entry.document_text:
            # Load PDF content from JSON chunks
            logger.info("Loading PDF content from JSON chunks")

            if entry.chunks_json:
                json_path = entry.chunks_json
            
            elif json_directory:
                # Ensure filename doesn't have .pdf extension for the JSON path
                file_base = entry.file_name.lower()
                json_path = os.path.join(json_directory, f"{file_base}_output.json")
            
            else:
                logger.error("Path to JSON directory with PDF text chunks required for information extraction")
                raise ValueError("Path to JSON directory with PDF text chunks required for information extraction")
                
            try:
                # Get PDF context and update the entry
                pdf_context = get_pdf_context_from_json(
                    json_path = json_path, 
                    pages = entry.page_number,
                    max_tokens = getattr(self.llm_handler, 'token_limit', 100000)
                    )
                entry.update_field('document_text', pdf_context)
                logger.info(f"Loaded {len(pdf_context)} characters of text from {json_path}")
            except FileNotFoundError:
                logger.error(f"JSON file for {entry.file_name} not found at {json_path}")
                return None, None
            except Exception as e:
                logger.error(f"Error loading PDF content: {str(e)}")
                return None, None

            
        else:
            logger.info("Benchmark entry already has relevant document text to extract information")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Successfully formatted prompt for information extraction (length: {len(prompt)})")
            return prompt, None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
            self, 
            entry: IEBenchmarkEntry,
            **kwargs
            ) -> Dict[str, Any]:
        """
        Evaluate a single information extraction task benchmark entry.
        
        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            json_directory: Directory with JSON chunks.
            max_attempts: Max attempts to generate a valid response.
            retry_delay: Delay in seconds for retrying prompt
            
        Returns:
            Dictionary with extracted information results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        
        logger.info(f"Evaluating information extraction entry with ID: {entry.entry_id}")

        # Create prompt for the entry
        prompt,_ = self.prepare_prompt(entry, **kwargs)

        # Return None if no prompt is created
        if not prompt:
            return None
        
        # Use the response processor to extract information with retries
        clean_response = self.process_ie_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            field_type=entry.field_type,
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )
        
        # If no valid response was generated
        if clean_response is None:
            logger.warning(f"Failed to extract information for entry {entry.entry_id} after {max_attempts} attempts")
            return None
        
        # Create result dictionary
        result = {
            "field_name": entry.field_name,
            "answer_expected": entry.answer,
            "answer_predicted": clean_response
        }

        # Add optional fields if they exist
        for field in ['file_name', 'page_number', 'entry_id']:
            if hasattr(entry, field) and getattr(entry, field) is not None:
                result[field] = getattr(entry, field)
            
        logger.info(f"Information extraction task completed for entry {entry.entry_id}")
        
        return result
    
    def process_ie_response_with_retries(
        self,
        get_response_func: Callable[[],str],
        field_type: str = "string",
        max_attempts: Optional[int] = None,
        retry_delay: Optional[int] = None
        ) -> Any:
        """
        Process an information extraction response with retries, applying field type-specific validation.
        
        Args:
            get_response_func: Function to get the response (no arguments).
            field_type (str): Type of field being extracted ('string', 'integer', 'date', 'array_string').
            max_attempts (Optional[int]): Maximum number of attempts, defaults to self.max_retries.
            retry_delay (Optional[int]): Duration in seconds between retry attempts
            
        Returns:
            Any: The validated and properly formatted response or None if validation always fails.
        """
        
        # Process with retries
        for attempt in range(1, max_attempts+1):
            try:
                logger.info(f"Attempt {attempt}/{max_attempts} to generate IE response for {field_type} field")
                
                # Get raw response
                response = get_response_func().strip()
                cleaned_response = response.strip()
                
                # Validate and format based on field type
                is_valid, formatted_value = validate_extracted_information(cleaned_response, field_type)
                
                if is_valid:
                    logger.info(f"Valid {field_type} response generated on attempt {attempt}")
                    return formatted_value
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {str(e)}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
        
        logger.warning(f"Failed to generate valid {field_type} response after {max_attempts} attempts")
        return None
    
    def extract_response(self, 
            entry: IEBenchmarkEntry, 
            response: str
            ) -> Optional[Dict[str, Any]]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (IEBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Optional[Dict[str, Any]]): Dictionary of processed result if valid, otherwise None
        """
        # Validate and extract information using response processor
        is_valid, extracted_response = validate_extracted_information(
            response, 
            field_type=entry.field_type
        )
        
        if is_valid:
            # Create result dictionary
            result = {
                "field_name": entry.field_name,
                "answer_expected": entry.answer,
                "answer_predicted": extracted_response
            }
            
            # Add optional fields if they exist
            for field in ['file_name', 'page_number', 'entry_id']:
                if hasattr(entry, field) and getattr(entry, field) is not None:
                    result[field] = getattr(entry, field)
            
            return result
            
        else:
            return None
                    

    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('configs/ie_cx_gemini.yaml')
    
    from utils import load_benchmark_entries
    entries = load_benchmark_entries(conf)

    from dotenv import load_dotenv
    load_dotenv(".env")
    
    # Load the GoogleGenAI model
    from llm_handlers.vertex_gemini_handler import GoogleGenAIHandler
    genai_client = GoogleGenAIHandler(model_name="gemini-2.5-pro")
    
    # IE evaluator agent check
    eval_config = conf["evaluation"]
    eval_client = InformationExtractor(genai_client, eval_config["prompt_file"])
    
    response = eval_client.evaluate_entry(
        entries[0], 
        **eval_config["eval_kwargs"]
        )
    logger.info(response)