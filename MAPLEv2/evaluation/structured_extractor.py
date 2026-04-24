import os
import sys
from typing import Dict, Any, Optional, Callable, Tuple, List, Type, Union
import time
from pydantic import BaseModel
import re, json

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils import StructuredIEBenchmarkEntry
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager
from utils import HierarchicalPydanticGenerator
from utils import generate_instruction, replace_nulls_with_defaults

if __name__ == "__main__":
    LogManager.initialize("logs/test_sie_evaluator.log")

logger = LogManager.get_logger("sie_extractor")


class StructuredExtractor(BaseEvaluator):
    """
    Evaluator for metadata-extraction benchmarks.
    """

    def __init__(self, llm_handler, prompt_template_source, is_template_file = True):
        super().__init__(llm_handler, prompt_template_source, is_template_file)

        # Initialize the Hierarchical Pydantic Generator
        self.schema_generator = HierarchicalPydanticGenerator()

    def prepare_prompt(
            self, 
            entry:StructuredIEBenchmarkEntry, 
            **kwargs
            ) -> Tuple[str, Optional[List[str]]]:
        if not entry.instruction_block:
            instructions = generate_instruction(entry.schema_json)
            entry.update_field('instruction_block', instructions)
            logger.info(f"Created instructions for structured data extraction (length: {len(instructions)})")
        
        # Format the prompt
        try:
            prompt = self.prompt_manager.format_prompt(entry)
            logger.info(f"Formatted prompt for structured data extraction (length: {len(prompt)})")
            return prompt, None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None, None

    def evaluate_entry(
            self, 
            entry: StructuredIEBenchmarkEntry,
            **kwargs
            ) -> Dict[str, Any]:
        """
        Evaluate a single tribe extraction task benchmark entry.
        
        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            max_attempts: Max attempts to generate a valid response.
            response_format (Optional[pydantic.BaseModel]): Format in which response is required
            
        Returns:
            Dictionary with extracted metadata results.
        """
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        logger.info(f"Evaluating structured extraction benchmark entry with ID: {entry.entry_id}")

        # Load the schema JSON file
        pydantic_code = self.schema_generator.generate_from_json_file(json_file=entry.schema_json)
        ResponseSchema = self.schema_generator.exec_with_globals(pydantic_code, "Output")

        # Get the prompt
        prompt, _ = self.prepare_prompt(entry, **kwargs)

        # If prompt is None, return None result
        if not prompt:
            return None
        
        # Use the response processor to extract metadata with retries
        clean_response = self.process_sie_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt, response_format=ResponseSchema),
            response_format=ResponseSchema,
            max_attempts=max_attempts,
            retry_delay=retry_delay
        )
        
        # If no valid response was generated
        if clean_response is None:
            logger.warning(f"Failed to extract structured information for entry {entry.entry_id} after {max_attempts} attempts")
            return None
        
        # Create result dictionary
        result = {
            "entry_id": entry.entry_id,
            "answer_expected": entry.info,
            "answer_predicted": clean_response
        }
        
        logger.info(f"Structured extraction task completed for entry {entry.entry_id}")
        
        return result
    
    def process_sie_response_with_retries(
        self,
        get_response_func: Callable[[],Union[str,Type[BaseModel]]],
        response_format: Type[BaseModel],
        max_attempts: Optional[int] = None,
        retry_delay: Optional[int] = None
        ) -> Any:
        """
        Process an information extraction response with retries, applying field type-specific validation.
        
        Args:
            get_response_func: Function to get the response (no arguments).
            response_format (pydantic.BaseModel): pydantic schema to return response.
            max_attempts (Optional[int]): Maximum number of attempts, defaults to self.max_retries.
            retry_delay (Optional[int]): Duration in seconds between retry attempts
            
        Returns:
            Any: The validated and properly formatted response or None if validation always fails.
        """
        
        # Process with retries
        for attempt in range(1, max_attempts+1):
            try:
                logger.info(f"Attempt {attempt}/{max_attempts} to generate structured IE response")
                
                # Get raw response
                response = get_response_func()
                
                # Handle returned response
                if isinstance(response, str):
                    # handle string response
                    try:
                        # Pattern to match the list inside code blocks
                        pattern = r'```(?:python|json)?\s*(\{.*?\})\s*```'
                        match = re.search(pattern, response, re.DOTALL)
                        if match:
                            dict_string = match.group(1)
                            try:
                                formatted_response = json.loads(dict_string)
                                validated_response = response_format(**formatted_response)
                                if isinstance(formatted_response, dict):
                                    formatted_response = replace_nulls_with_defaults(formatted_response, response_format)
                                return formatted_response
                            
                            except json.JSONDecodeError as e:
                                logger.error(f"Error in attempt {attempt}: Failed to convert matched dictionary object to dictionary using JSON decoder: {str(e)}")
                            
                            except Exception as e:
                                logger.error(f"Error in attempt {attempt}: Failed to validate response to response format: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error in attempt {attempt}: Failed to convert response to JSON format: {str(e)}")
                
                elif isinstance(response, response_format):
                    formatted_response = response.model_dump()
                    validated_response = response_format(**formatted_response)
                    if isinstance(formatted_response, dict):
                        formatted_response = replace_nulls_with_defaults(formatted_response, response_format)
                    return formatted_response
                
                else:
                    logger.warning(f"Unknown type of response returned by LLM handler")
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: Failed to validate response to provided schema: {str(e)}")
            
            if attempt < max_attempts:
                logger.info(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
        
        logger.warning(f"Failed to generate valid response after {max_attempts} attempts")
        return None
    
    def extract_response(
            self, 
            entry: StructuredIEBenchmarkEntry, 
            response: str
            ):
        # Load the schema JSON file
        pydantic_code = self.schema_generator.generate_from_json_file(json_file=entry.schema_json)
        ResponseSchema = self.schema_generator.exec_with_globals(pydantic_code, "Output")

        try:
            formatted_response = json.loads(response)
            validated_response = ResponseSchema(**formatted_response)
            
            # Create result dictionary
            result = {
                "entry_id": entry.entry_id,
                "answer_expected": entry.info,
                "answer_predicted": formatted_response
            }
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to convert matched dictionary object for {entry.entry_id} to dictionary using JSON decoder: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to validate response for {entry.entry_id} to response format: {str(e)}")
            return None

    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    
    from utils.dataloader import load_benchmark_entries_from_json
    entries = load_benchmark_entries_from_json(
        "input/benchmark/fedreg-data.json", 
        StructuredIEBenchmarkEntry, "structured extraction", 
        field_mapping={'text':'text','info':'info', 'schema_json':'schema_json'}
        )

    from llm_handlers import AzureOpenAIHandler
    llm_client = AzureOpenAIHandler(
        model_name="gpt-4-turbo-preview", token_limit=100000,
        )
    
    prompt_file = "input/prompts/sie_prompt.txt"
    eval_client = StructuredExtractor(llm_client, prompt_file, is_template_file=True)
    response = eval_client.evaluate_entry(entries[0], max_attempts=3)
    
    import json
    with open("results/test_fedreg.json", 'w') as jsonfile:
        json.dump(response['answer_predicted'], jsonfile, indent=2)