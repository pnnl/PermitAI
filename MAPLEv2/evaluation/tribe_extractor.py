import os
import sys
from typing import List, Dict, Any, Callable, Optional

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import TribalBenchmarkEntry
from utils.parser_utils import extract_section_by_name, extract_section_by_number
from utils.pdf_utils import count_tokens, truncate_text
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager
from utils import validate_extracted_information
import time

if __name__ == "__main__":
    LogManager.initialize("logs/test_tribedata_evaluator.log")

logger = LogManager.get_logger("tribe_extractor")


class TribeExtractor(BaseEvaluator):
    """
    Evaluator for metadata-extraction benchmarks.
    """

    def prepare_prompt(self, entry:TribalBenchmarkEntry, **kwargs):
        # Check if entry has attribute 'document_text'
        if not entry.document_text:
            # Load PDF content from JSON chunks
            logger.info("Loading PDF content from selected sections")

            if entry.source_type == 'name':
                try:
                    pdf_context = extract_section_by_name(pdf_path=entry.file_name, section_name=entry.source_caption)
                    entry.update_field('document_text', pdf_context)
                    logger.info(f"Loaded {len(pdf_context)} characters of text from section name '{entry.source_caption}' of file path {entry.file_name}")
                except FileNotFoundError:
                    logger.error(f"PDF file {entry.file_name} not found")
                    return None, None
                except Exception as e:
                    logger.error(f"Error loading PDF content: {str(e)}")
                    return None, None
            
            elif entry.source_type == 'number':
                try:
                    pdf_context = extract_section_by_number(pdf_path=entry.file_name, section_number=entry.source_caption)
                    entry.update_field('document_text', pdf_context)
                    logger.info(f"Loaded {len(pdf_context)} characters of text from section no. {entry.source_caption} of file path {entry.file_name}")
                except FileNotFoundError:
                    logger.error(f"PDF file {entry.file_name} not found")
                    return None, None
                except Exception as e:
                    logger.error(f"Error loading PDF content: {str(e)}")
                    return None, None
            
            else:
                logger.error("Invalid source type. Accepted source type is section 'name' or section 'number'")
                raise ValueError("Invalid source type. Accepted source type is section 'name' or section 'number'")
            
        else:
            logger.info("Benchmark entry already has relevant document text to extract metadata")

        # Truncate the input text if it exceeds the max_tokens limit
        if entry.document_text:
            max_tokens = self.llm_handler.token_limit
            token_count = count_tokens(pdf_context)
            logger.info(f"Current token count = {token_count}")
            if token_count > max_tokens:
                logger.info(f"Truncating the context because it exceeds maximum token count {max_tokens}")
                pdf_context = truncate_text(pdf_context, max_tokens)
            
            # Update with truncated text
            entry.update_field('document_text', pdf_context)
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for metadata extraction (length: {len(prompt)})")
            return prompt, None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
            self, 
            entry: TribalBenchmarkEntry,
            **kwargs
            ) -> Dict[str, Any]:
        """
        Evaluate a single tribe extraction task benchmark entry.
        
        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            max_attempts: Max attempts to generate a valid response.
            
        Returns:
            Dictionary with extracted metadata results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating metadata entry with ID: {entry.entry_id}")

        # Get the prompt
        prompt, _ = self.prepare_prompt(entry, **kwargs)

        # Return None if no prompt is created
        if not prompt:
            return None
        
        # Use the response processor to extract metadata with retries
        clean_response = self.process_ie_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            field_type="array_string",
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )
        
        # If no valid response was generated
        if clean_response is None:
            logger.warning(f"Failed to extract tribe data for entry {entry.entry_id} after {max_attempts} attempts")
            return None
        
        # Create result dictionary
        result = {
            "file_name": entry.file_name,
            "source_type": entry.source_type,
            "source": entry.source_caption,
            "answer_expected": entry.tribes,
            "answer_predicted": clean_response,
            "entry_id": entry.entry_id
        }
        
        logger.info(f"Tribe extraction task completed for entry {entry.entry_id}")
        
        return result
    
    def process_ie_response_with_retries(
        self,
        get_response_func: Callable[[],str],
        field_type: str = "array_string",
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
            entry: TribalBenchmarkEntry, 
            response: str
            ) -> Optional[Dict[str, Any]]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entry (TribalBenchmarkEntry): Benchmark entry that is being evaluated
            response (str): Raw response from batch job
            
        Returns:
            result (Optional[Dict[str, Any]]): Dictionary of processed result if valid, otherwise None
        """ 
        # Validate and extract information using response processor
        is_valid, clean_response = validate_extracted_information(
            response, 
            field_type="array_string"
        )
        
        if is_valid:
            # Create result dictionary
            result = {
                "file_name": entry.file_name,
                "source_type": entry.source_type,
                "source": entry.source_caption,
                "answer_expected": entry.tribes,
                "answer_predicted": clean_response,
                "entry_id": entry.entry_id
            }
            
            return result
            
        else:
            logger.warning(f"Invalid response for entry {entry.entry_id}: {response}")
            return None
                    
            
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('configs/tribe_mistral.yaml')
    
    from utils import load_benchmark_entries
    entries = load_benchmark_entries(conf)

    
    # Load the AWS Bedrock hosted model
    provider = conf["model"]["provider"]
    model_name = conf["model"]["name"]
    max_tokens = conf["model"]["max_tokens"] if "max_tokens" in conf["model"] else 20000
    from llm_handlers.aws_bedrock_handler import AWSBedrockHandler
    llm_client = AWSBedrockHandler(
        model_name, token_limit=max_tokens)
    
    # QA evaluator agent check
    eval_config = conf["evaluation"]
    eval_client = TribeExtractor(llm_client, eval_config["prompt_file"], is_template_file=True)
    response = eval_client.evaluate_entry(
        entries[0], 
        **eval_config["eval_kwargs"]
        )
    
    logger.info(response)