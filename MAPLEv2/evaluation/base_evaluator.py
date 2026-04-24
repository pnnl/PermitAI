from abc import ABC, abstractmethod
import os
import sys
import json
from typing import List, Any, Optional, Union, Dict, Tuple

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager
if __name__ == "__main__":
    LogManager.initialize("logs/test_base_evaluator.log")

logger = LogManager.get_logger(__name__)

from llm_handlers import BaseLLMHandler
from utils import PromptManager
from utils import pydantic_to_vertex_schema
from utils import HierarchicalPydanticGenerator
from utils import (
    QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
    BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
    MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry
    )

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluating benchmark entries using an LLM.
    
    This class defines the interface for LLM-based evaluators. Subclasses should
    implement the evaluate_entry method to handle specific benchmark entry types.
    """
    
    def __init__(self, 
                llm_handler: BaseLLMHandler, 
                prompt_template_source: str,
                is_template_file: bool = True
                ):
        """
        Initialize the BaseEvaluator.
        
        Args:
            llm_handler (BaseLLMHandler): Handler for interacting with the LLM.
            prompt_template_source (str): Either a path to the prompt template file or the template string itself.
            is_template_file (bool): If True, prompt_template_source is treated as a file path. 
                                   If False, prompt_template_source is treated as the template string.
        """
        self.llm_handler = llm_handler
        
        # Initialize the prompt manager based on the template source type
        if is_template_file:
            self.prompt_manager = PromptManager.from_file(prompt_template_source)
            logger.info(f"Initialized BaseEvaluator with prompt template file: {prompt_template_source}")
        else:
            self.prompt_manager = PromptManager.from_string(prompt_template_source)
            logger.info(f"Initialized BaseEvaluator with prompt template string (length: {len(prompt_template_source)})")
        
        logger.info(f"Prompt placeholders: {', '.join(self.prompt_manager.placeholders)}")
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to the output file.
        
        Args:
            results (Dict[str, Any]): Results to save.
            output_path (str): Path to the output file.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2)
            logger.info(f"Successfully saved results for {len(results)} entries to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _load_existing_results(self, output_path: str) -> Dict[str, Any]:
        """
        Load existing results from the output file if it exists.
        
        Args:
            output_path (str): Path to the output file.
            
        Returns:
            Dict[str, Any]: Dictionary with entry_id as keys and evaluation results as values.
        """
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    logger.info(f"Loaded existing results for {len(data)} entries from {output_path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load existing results from {output_path}: {str(e)}")
                return {}
        else:
            logger.info(f"No existing results file found at {output_path}, creating new one")
            return {}
    
    def _append_result(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Append a single result to the output file.
        
        Args:
            results (Dict[str, Any]): Results dictionary to append to.
            output_path (str): Path to the output file.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2)
            logger.info(f"Updated results at {output_path}")
        except Exception as e:
            logger.error(f"Error appending result: {str(e)}")
            raise
    
    @abstractmethod
    def evaluate_entry(self, 
                      entry: Union[
                            QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                            MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry
                        ], 
                      **kwargs) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single benchmark entry.
        
        This abstract method should be implemented by subclasses to handle
        specific benchmark entry types.
        
        Args:
            entry (Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry]): The benchmark entry to evaluate.
            **kwargs: Additional keyword arguments for specific evaluation types.
                max_attempts (int): Maximum number of attempts to generate a valid response.
                retry_delay (int): Delay between retry attempts in seconds.
            
        Returns:
            Optional[Dict[str, Any]]: The evaluation results, or None if evaluation failed.
        """
        pass

    @abstractmethod
    def prepare_prompt(
        entry: Union[
            QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
            MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry
        ], 
        **kwargs
        ) -> Tuple[str, Optional[List[str]]]:
        """
        Evaluate a single benchmark entry.
        
        This abstract method should be implemented by subclasses to handle
        specific benchmark entry types.
        
        Args:
            entry (Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, 
            StructuredIEBenchmarkEntry, BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, 
            CommentClassificationBenchmarkEntry,MapClassifyBenchmarkEntry]): The benchmark entry to evaluate.
            
            **kwargs: Additional keyword arguments for specific evaluation types.
            
        Returns:
            Tuple[str, Optional[List[str]]]: A tuple of (text_prompt, list_of_image_paths).
            The image_paths list can be None or empty list if no images are needed.
        """
        pass
    
    def evaluate_batch(self,
                      entries: List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                        BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                        MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry]],
                      output_path: str,
                      continue_from_previous: bool = True,
                      **kwargs) -> str:
        """
        Evaluate multiple benchmark entries and save results to a single JSON file.
        
        This method handles the batch processing of entries, calling the abstract
        evaluate_entry method for each entry to be processed. It manages loading
        and saving of results to enable resuming batch processing.
        
        Args:
            entries (List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
            MapClassifyBenchmarkEntry]]): The benchmark entries to evaluate.
            output_path (str): Path to save the results JSON file.
            continue_from_previous (bool): Whether to continue from previous evaluation results.
            **kwargs: Additional keyword arguments to pass to evaluate_entry().
            
        Returns:
            str: Path to the saved results file.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Evaluating batch of {len(entries)} entries (continue_from_previous={continue_from_previous})")
        
        # Initialize or load results data
        results_data = {}
        if continue_from_previous:
            results_data = self._load_existing_results(output_path)
        
        # Determine which entries to evaluate
        entries_to_evaluate = []
        if continue_from_previous:
            missing_id_count = 0
            new_entry_count = 0
            skipped_entry_count = 0
            
            for entry in entries:
                entry_id = getattr(entry, 'entry_id', None)
                if entry_id is None:
                    missing_id_count += 1
                    # Skip entries with missing IDs (don't add to evaluation list)
                else:
                    if entry_id not in results_data:
                        new_entry_count += 1
                        entries_to_evaluate.append(entry)
                    else:
                        skipped_entry_count += 1
            
            # Log summary of entries
            logger.info(f"Entry summary: {new_entry_count} new entries to evaluate, "
                        f"{skipped_entry_count} already evaluated entries skipped")
            
            if missing_id_count > 0:
                logger.warning(f"Skipped {missing_id_count} entries missing identifier (entry_id)")
        else:
            # Check for entries with missing IDs
            valid_entries = []
            missing_id_count = 0
            
            for entry in entries:
                entry_id = getattr(entry, 'entry_id', None)
                if entry_id is None:
                    missing_id_count += 1
                    # Skip entries with missing IDs
                else:
                    valid_entries.append(entry)
            
            entries_to_evaluate = valid_entries
            logger.info(f"Processing {len(entries_to_evaluate)} valid entries (continue_from_previous=False)")
            
            if missing_id_count > 0:
                logger.warning(f"Skipped {missing_id_count} entries missing identifier (entry_id)")
        
        logger.info(f"Will evaluate {len(entries_to_evaluate)} entries out of {len(entries)} total")
        
        # Process each entry
        for i, entry in enumerate(entries_to_evaluate):
            entry_id = getattr(entry, 'entry_id', None)
            logger.info(f"Processing entry {i+1}/{len(entries_to_evaluate)}: {entry_id}")
            
            result = self.evaluate_entry(entry, **kwargs)
            
            # Only add valid results to results_data
            if result is not None:
                results_data[entry_id] = result
                # Save after each valid entry to prevent data loss
                self._append_result(results_data, output_path)
            else:
                logger.warning(f"Skipping invalid result for entry: {entry_id}")
        
        logger.info(f"Completed evaluation of {len(entries_to_evaluate)} entries")
        logger.info(f"Total entries in results file: {len(results_data)}")
        return output_path
    
    def _identify_unevaluated_entries(self, entries: List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                        BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                        MapClassifyBenchmarkEntry,CommentDelineateBenchmarkEntry]], 
                        output_path: str) -> List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                        BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                        MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry]]:
        """
        Identify which entries have not been evaluated yet by checking existing results.
        
        Args:
            entries: List of benchmark entries
            output_path: Path to the results JSON file
            
        Returns:
            List of entries that need to be evaluated
        """
        # Load existing results
        existing_results = self._load_existing_results(output_path)
        
        unevaluated_entries = []
        missing_id_count = 0
        
        for entry in entries:
            entry_id = getattr(entry, 'entry_id', None)
            if entry_id is None:
                missing_id_count += 1
                logger.warning(f"Entry missing entry_id, skipping: {entry}")
                continue
                
            if entry_id not in existing_results:
                unevaluated_entries.append(entry)
        
        logger.info(f"Found {len(unevaluated_entries)} unevaluated entries out of {len(entries)} total entries")
        if missing_id_count > 0:
            logger.warning(f"Skipped {missing_id_count} entries missing entry_id")
        
        return unevaluated_entries

    def _create_batch_prompts(
            self, 
            entries: List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                            MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry]], 
            **kwargs
            ) -> List[Dict[str, Any]]:
        """
        Create prompts for batch submission using the prepare_prompt method.
        
        Args:
            entries: List of benchmark entries to create prompts for
            **kwargs: Additional arguments for prompt preparation
            
        Returns:
            List of prompt dictionaries in the format expected by batch API
        """
        prompt_data = []
        failed_entries = []
        
        for entry in entries:
            try:
                # Get text prompt and image paths (always returns tuple now)
                text_prompt, image_paths = self.prepare_prompt(entry, **kwargs)
                
                if text_prompt is None:
                    failed_entries.append(entry.entry_id)
                    logger.warning(f"Failed to create prompt for entry: {entry.entry_id}")
                    continue

                # Create parts list starting with text
                parts = [{"text": text_prompt}]
                    
                # Add image parts if present
                if image_paths:
                    for image_path in image_paths:
                        if image_path and os.path.exists(image_path):
                            parts.append({
                                "inline_data": {
                                    "mime_type": self._get_mime_type(image_path),
                                    "data": image_path  # Store path for now, will be replaced with base64 data
                                }
                            })
                        else:
                            logger.warning(f"Image not found: {image_path} for entry: {entry.entry_id}")
                
                # Format for Gemini batch API
                prompt_dict = {
                    "id": entry.entry_id,
                    "request": {
                        "contents": [
                            {
                                "role": "user", 
                                "parts": parts
                            }
                        ]
                    }
                }

                # add response format
                if kwargs.get('structured_response', False):
                    if not hasattr(entry, 'schema_json'):
                        logger.warning(f"Benchmark entry class {entry.__class__} does not have attribute for JSON schema. Cannot produce structured output.")
                    
                    else:
                        schema_generator = HierarchicalPydanticGenerator()
                        pydantic_code = schema_generator.generate_from_json_file(json_file=entry.schema_json)
                        ResponseSchema = schema_generator.exec_with_globals(pydantic_code, "Output")
                        schema = pydantic_to_vertex_schema(ResponseSchema)
                        prompt_dict["request"]["generationConfig"] = {
                            "responseMimeType": "application/json", 
                            "responseSchema" : schema
                        }
                        logger.info(f"Successfully added response format from schema JSON {entry.schema_json}")
                
                prompt_data.append(prompt_dict)
                
            except Exception as e:
                failed_entries.append(getattr(entry, 'entry_id', 'unknown'))
                logger.error(f"Error creating prompt for entry {getattr(entry, 'entry_id', 'unknown')}: {str(e)}")
        
        logger.info(f"Created {len(prompt_data)} prompts for batch submission")
        if failed_entries:
            logger.warning(f"Failed to create prompts for {len(failed_entries)} entries: {failed_entries}")
        
        return prompt_data
    
    def _process_images_in_prompts(self, prompt_data: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Process images in prompts by converting to base64 or uploading to GCS.
        
        Args:
            prompt_data: List of prompt dictionaries
            **kwargs: Additional arguments (bucket_name, upload_image_path)
            
        Returns:
            List of processed prompt dictionaries
        """
        bucket_name = kwargs.get('bucket_name', 'maple_evaluations')
        upload_image_path = kwargs.get('upload_image_path', None)
        
        processed_data = []
        
        for prompt_dict in prompt_data:
            prompt_id = prompt_dict["id"]
            processed_dict = json.loads(json.dumps(prompt_dict))  # Deep copy
            
            # Process parts in the request
            for content in processed_dict["request"]["contents"]:
                for part in content["parts"]:
                    if "inline_data" in part:
                        image_path = part["inline_data"]["data"]
                        
                        if upload_image_path:
                            # Upload to GCS and use URI
                            gcs_uri = self.llm_handler.upload_image_to_gcs(image_path, bucket_name, upload_image_path, prompt_id)
                            # Convert to GCS file data format
                            part["file_data"] = {
                                "mime_type": part["inline_data"]["mime_type"],
                                "file_uri": gcs_uri
                            }
                            del part["inline_data"]
                        else:
                            # Convert to base64
                            base64_data = self._image_to_base64(image_path)
                            part["inline_data"]["data"] = base64_data
            
            processed_data.append(processed_dict)
        
        return processed_data

    def _get_mime_type(self, image_path: str) -> str:
        """
        Get MIME type for image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string
        """
        extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(extension, 'image/jpeg')
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            import base64
            
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"Converted image {image_path} to base64 (length: {len(encoded_string)})")
            return encoded_string
            
        except Exception as e:
            logger.error(f"Error converting image {image_path} to base64: {str(e)}")
            raise

    def _save_prompts_to_jsonl(self, prompt_data: List[Dict[str, Any]], jsonl_path: str, **kwargs) -> None:
        """
        Save prompt data to JSONL file for batch submission.
        
        Args:
            prompt_data: List of prompt dictionaries
            jsonl_path: Path to save the JSONL file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

            # Process images and convert to base64 or GCS URIs
            processed_prompt_data = self._process_images_in_prompts(prompt_data, **kwargs)
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for prompt_dict in processed_prompt_data:
                    json_line = json.dumps(prompt_dict)
                    f.write(json_line + "\n")
            
            logger.info(f"Saved {len(prompt_data)} prompts to {jsonl_path}")
            
        except Exception as e:
            logger.error(f"Error saving prompts to JSONL: {str(e)}")
            raise

    def _submit_batch_job(self, jsonl_path: str, **kwargs) -> Dict[str, Any]:
        """
        Submit batch job using the LLM handler and return responses.
        
        Args:
            jsonl_path: Path to the input JSONL file
            **kwargs: Batch job configuration parameters
            
        Returns:
            Dictionary of responses from batch job
        """
        # Validate LLM handler supports batch jobs
        if not hasattr(self.llm_handler, 'batch_job'):
            raise ValueError(f"LLM handler {self.llm_handler.__class__.__name__} does not support batch jobs")
        
        # Extract batch job parameters
        bucket_name = kwargs.get('bucket_name', 'maple_evaluations')
        upload_file_path = kwargs.get('upload_file_path', 'input_data/prompts.jsonl')
        predictions_file_path = kwargs.get('predictions_file_path', 'responses')
        batch_response_path = kwargs.get('batch_response_path', 'batch_responses.json')
        
        try:
            logger.info(f"Submitting batch job with {jsonl_path}")
            responses = self.llm_handler.batch_job(
                ip_file_path=jsonl_path,
                upload_file_path=upload_file_path,
                op_file_path=predictions_file_path,
                bucket_name=bucket_name
            )
            
            # Save raw responses to file
            try:
                # Ensure directory exists for batch response file
                os.makedirs(os.path.dirname(batch_response_path), exist_ok=True)
                
                with open(batch_response_path, 'w', encoding='utf-8') as f:
                    json.dump(responses, f, indent=2)
                
                logger.info(f"Saved raw batch responses to {batch_response_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save batch responses to file: {str(e)}")
                # Continue execution even if saving fails
            
            logger.info(f"Batch job completed successfully with {len(responses)} responses")
            return responses
            
        except Exception as e:
            logger.error(f"Error submitting batch job: {str(e)}")
            raise

    def _append_batch_results_to_file(self, new_results: Dict[str, Any], output_path: str) -> None:
        """
        Append new batch results to the existing results file.
        
        Args:
            new_results: New results to append
            output_path: Path to the results JSON file
        """
        try:
            # Load existing results
            existing_results = self._load_existing_results(output_path)
            
            # Merge new results
            existing_results.update(new_results)
            
            # Save merged results
            self._save_results(existing_results, output_path)
            
            logger.info(f"Appended {len(new_results)} new results to {output_path}")
            logger.info(f"Total entries in results file: {len(existing_results)}")
            
        except Exception as e:
            logger.error(f"Error appending batch results to file: {str(e)}")
            raise

    def execute_gemini_batch_job(
            self, 
            entries: List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
                BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
                MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry]], 
            output_path: str,
            continue_from_previous: bool = True,
            **kwargs
            ) -> str:
        """
        Execute complete batch job workflow: identify unevaluated entries, 
        create prompts, submit batch job, extract responses, and save results.
        
        Args:
            entries: List of benchmark entries
            output_path: Path to save/append results
            continue_from_previous: Whether to continue from existing results or start fresh
            
            **kwargs: Configuration parameters for batch job
            Additional kwargs for image support:
                upload_images_to_gcs (bool): Whether to upload images to GCS (default: False, uses base64)
                bucket_name (str): GCS bucket name for image uploads
            Additional kwargs for structured response:
                structured_response (bool): flag to indicate structured response based on schema in entry
            
        Returns:
            Path to the results file
        """
        # Validate LLM handler
        if not hasattr(self.llm_handler, 'batch_job'):
            logger.error(f"Batch job execution is only available for handlers with batch_job method")
            raise ValueError(f"LLM handler {self.llm_handler.__class__.__name__} does not support batch jobs")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Starting batch job execution for {len(entries)} entries (continue_from_previous={continue_from_previous})")
        
        # Step 1: Determine which entries to evaluate
        if continue_from_previous:
            # Identify unevaluated entries by checking existing results
            entries_to_evaluate = self._identify_unevaluated_entries(entries, output_path)
        else:
            # Process all entries, ignoring existing results
            entries_to_evaluate = []
            missing_id_count = 0
            
            for entry in entries:
                entry_id = getattr(entry, 'entry_id', None)
                if entry_id is None:
                    missing_id_count += 1
                    logger.warning(f"Entry missing entry_id, skipping: {entry}")
                    continue
                entries_to_evaluate.append(entry)
            
            logger.info(f"Processing all {len(entries_to_evaluate)} valid entries (continue_from_previous=False)")
            if missing_id_count > 0:
                logger.warning(f"Skipped {missing_id_count} entries missing entry_id")
        
        if not entries_to_evaluate:
            if continue_from_previous:
                logger.info("No unevaluated entries found. Batch job execution complete.")
            else:
                logger.info("No valid entries to process. Batch job execution complete.")
            return output_path
        
        # Step 2: Create prompts for entries to evaluate
        prompt_data = self._create_batch_prompts(entries_to_evaluate, **kwargs)
        
        if not prompt_data:
            logger.warning("No valid prompts created. Batch job execution terminated.")
            return output_path
        
        # Step 3: Save prompts to JSONL file
        prompt_jsonl_path = kwargs.get('prompt_jsonl_path', 'batch_prompts.jsonl')
        self._save_prompts_to_jsonl(prompt_data, prompt_jsonl_path, **kwargs)
        
        # Step 4: Submit batch job
        try:
            raw_responses = self._submit_batch_job(prompt_jsonl_path, **kwargs)
        except Exception as e:
            logger.error(f"Batch job submission failed: {str(e)}")
            raise
        
        # Step 5: Extract and process responses (contains abstract method for each response)
        try:
            processed_results = self.extract_batch_responses(entries_to_evaluate, raw_responses)
        except Exception as e:
            logger.error(f"Error processing batch responses: {str(e)}")
            raise
        
        # Step 6: Save or append results to output file
        if processed_results:
            if continue_from_previous:
                # Append to existing results
                self._append_batch_results_to_file(processed_results, output_path)
            else:
                # Save as new results file (overwriting existing)
                self._save_results(processed_results, output_path)
                logger.info(f"Saved {len(processed_results)} new results to {output_path}")
        else:
            logger.warning("No valid results to save from batch job")
        
        logger.info(f"Batch job execution completed successfully. Results saved to {output_path}")
        return output_path
    
    @abstractmethod
    def extract_response(
        self, 
        entry: Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
            MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry], 
        response: str
        ) -> Dict[str,Any]:
        """
        Extract the actual response from each response in the batch job
        """
        pass

    def extract_batch_responses(
            self, 
            entries: List[Union[QABenchmarkEntry, IEBenchmarkEntry, TribalBenchmarkEntry, StructuredIEBenchmarkEntry,
            BinSummarizerBenchmarkEntry, CommentBinBenchmarkEntry, CommentClassificationBenchmarkEntry,
            MapClassifyBenchmarkEntry, CommentDelineateBenchmarkEntry]], 
            responses: Dict[str, Any]
            ) -> Dict[str, Any]:
        """
        Extract and process responses from batch job results.
        
        Args:
            entries: List of benchmark entries that were evaluated
            responses: Raw responses from batch job
            
        Returns:
            Dictionary of processed results with entry_id as keys
        """
        processed_results = {}
        failed_extractions = []
        
        for entry in entries:
            entry_id = entry.entry_id
            
            if entry_id not in responses:
                logger.warning(f"No response found for entry: {entry_id}")
                failed_extractions.append(entry_id)
                continue
            
            try:
                # Get the raw response
                raw_response = responses[entry_id]

                # Extract the acceptable response
                result = self.extract_response(entry, raw_response)
                
                # Check if the extracted result is non None type
                if result:
                    processed_results[entry_id] = result
                    logger.info(f"Successfully processed response for entry: {entry_id}")
                    
                else:
                    logger.warning(f"Invalid response for entry {entry_id}: {raw_response}")
                    failed_extractions.append(entry_id)
                    
            except Exception as e:
                logger.error(f"Error processing response for entry {entry_id}: {str(e)}")
                failed_extractions.append(entry_id)
        
        logger.info(f"Successfully processed {len(processed_results)} responses")
        if failed_extractions:
            logger.warning(f"Failed to process {len(failed_extractions)} responses: {failed_extractions}")
        
        return processed_results