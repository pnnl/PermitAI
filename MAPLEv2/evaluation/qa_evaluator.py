import os
import sys
from typing import List, Tuple, Dict, Any, Optional
import time
import re

from llama_index.core.chat_engine.types import AgentChatResponse

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import QABenchmarkEntry
from utils import setup_rag_chat_client
from utils import get_pdf_context_from_json
from evaluation.base_evaluator import BaseEvaluator
from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_qa_evaluator.log")

logger = LogManager.get_logger(__name__)


class QAEvaluator(BaseEvaluator):
    """
    Evaluator for question-answering benchmarks.
    Supports different context types for evaluation.
    """
    
    def _validate_context_requirements(self, entry: QABenchmarkEntry, context_type: str):
        """Validate that entry has required fields for the context type."""
        if context_type == "pdf" and not hasattr(entry, "file_name"):
            raise ValueError("Context type 'pdf' requires 'file_name' attribute")
            
        if context_type == "rag" and not hasattr(entry, "file_name"):
            raise ValueError("Context type 'rag' requires 'file_name' attribute")
            
        if context_type == "gold" and not hasattr(entry, "context"):
            raise ValueError("Context type 'gold' requires 'context' attribute")
        
    def prepare_prompt(self, entry:QABenchmarkEntry, **kwargs):
        """
        Prepares the prompt for a single benchmark entry
        """
        # Get context type
        context_type = kwargs.get('context_type', None)
        
        # Prepare the question
        question = entry.question
        if hasattr(entry, "question_type") and entry.question_type == "closed":
            question += " Return the answer in a single 'yes' or 'no' only. Do not provide additional explanations."
            entry.update_field("question", question)

        # Process based on context type
        if not context_type or context_type == "none":
            logger.info("Context type: NONE - Direct query without context")
            
            # Update context to empty string
            gold_context = entry.context
            entry.update_field('context', "")
            prompt = self.prompt_manager.format_prompt(entry)
            
            # Revert the entry context field to the old gold context
            entry.update_field('context', gold_context)
            return prompt, None
            
        elif context_type == "pdf":
            logger.info("Context type: PDF - Using entire PDF content")
            json_directory = kwargs.get('json_directory', None)
            
            if entry.chunks_json:
                json_path = entry.chunks_json 
            
            elif json_directory and entry.file_name:
                pdf_name = entry.file_name
                if pdf_name.endswith(".pdf"):
                    pdf_name = pdf_name[:-4]
                pdf_name = pdf_name.replace(" ", "_")
                json_path = f"{json_directory}{pdf_name}_output.json"
                
            else:
                logger.error("JSON directory and file_name required for PDF context")
                return None, None
            
            # Update the entry context with PDF context
            gold_context = entry.context
            pdf_context = get_pdf_context_from_json(json_path, max_tokens=self.llm_handler.token_limit)
            entry.update_field('context', pdf_context)
            prompt = self.prompt_manager.format_prompt(entry)

            # Revert the entry context field to the old gold context
            entry.update_field('context', gold_context)
            return prompt, None
            
        elif context_type == "gold":
            logger.info("Context type: GOLD - Using provided gold context")
            prompt = self.prompt_manager.format_prompt(entry)
            return prompt, None
                
        else:
            logger.error(f"Unsupported context type for 'prepare_prompt' method: {context_type}")
            return None, None
    
    def clean_qa_response(self, response: str, question_type: str) -> str:
        """
        Clean the QA response by removing system tokens and unnecessary whitespace.
        For closed questions, standardize to "yes", "no", or "unclear".
        
        Args:
            response (str): The raw response from the language model.
            question_type (str): The question type of the qa benchmark entry being evaluated.
        
        Returns:
            str: The cleaned response.
        """
        # Remove HTML-like decorators
        cleaned = re.sub(r'<<[^>]*>>', '', response)  # Remove opening tags
        cleaned = re.sub(r'<</[^>]*>>', '', cleaned)  # Remove closing tags
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if question_type == 'closed':
            # For closed questions, return only "yes" or "no"
            lower_cleaned = cleaned.lower()
            if 'not sure' in lower_cleaned or 'uncertain' in lower_cleaned or 'unclear' in lower_cleaned:
                return "unclear"
            elif 'yes' in lower_cleaned or 'true' in lower_cleaned or 'correct' in lower_cleaned:
                return "yes"
            elif 'no' in lower_cleaned or 'false' in lower_cleaned or 'incorrect' in lower_cleaned:
                return "no"
            else:
                logger.warning(f"Unclear response for closed question: {cleaned}")
                return "unclear"
        
        return cleaned
    
    def should_retry_qa(self, response: str, question_type: str) -> bool:
        """
        Determine if the QA response should be retried based on certain conditions.
        
        Args:
            response (str): The generated response.
            question_type (str): The question type of the qa benchmark entry being evaluated.
        
        Returns:
            bool: True if the response should be retried, False otherwise.
        """
        if not response or response == "Empty Response":
            logger.warning("Empty response received, will retry")
            return True
            
        if response.startswith("I'm sorry, but the context"):
            logger.warning("Response indicates context issues, will retry")
            return True
            
        if question_type == "closed":
            if not (response.lower().startswith("yes") or response.lower().startswith("no")):
                logger.warning("Closed question not answered in correct format")
                return True
                
        return False

    def process_qa_response_with_retries(
            self, 
            get_response_func, 
            question_type: str,
            max_attempts: Optional[int] = 3,
            retry_delay: Optional[int] = 10
            ) -> Any:
        """
        Process a response with retries using the provided functions.
        
        Args:
            get_response_func: Function to get the response (no arguments).
            question_type (str): Type of question.
            max_attempts (Optional[int]): Maximum number of attempts, defaults to 3.
            retry_delay (Optional[int]): Time delay in seconds, defaults to 10.
            
        Returns:
            Any: The valid response or None if validation always fails.
        """
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Attempt {attempt+1}/{max_attempts} to generate response")
                qa_response = get_response_func()

                # Clean the response
                cleaned_qa_response = self.clean_qa_response(qa_response, question_type)
                
                if not self.should_retry_qa(cleaned_qa_response, question_type):
                    logger.info(f"Valid response generated on attempt {attempt+1}")
                    return cleaned_qa_response
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}: {str(e)}")
            
            if attempt < max_attempts - 1:
                logger.info(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)
        
        logger.warning(f"Failed to generate valid response after {max_attempts} attempts")
        return None
    
    def _process_rag_query(
            self, 
            entry: QABenchmarkEntry, 
            chromadb_path: str, 
            collection_name: str,
            doc_id_csv_path: str,
            embed_model_name: str,
            max_attempts: int
            ) -> Tuple[str, List[str], List[str]]:
        """
        Process a query using RAG.
        
        Args:
            entry: The benchmark entry to evaluate.
            chromadb_path: Path to ChromaDB.
            collection_name: ChromaDB collection name.
            doc_id_csv_path: Path to CSV with document ID mappings.
            embed_model_name: Name of the embedding model to use.
            max_attempts: Maximum number of attempts.
            
        Returns:
            Tuple of (response, chunks, source_nodes).
        """
            
        # Set up the RAG chat client using the external utility function
        chat_client = setup_rag_chat_client(
            chromadb_path=chromadb_path, 
            collection_name=collection_name,
            embed_model_name=embed_model_name,
            llm_handler=self.llm_handler,
            doc_id_csv_path=doc_id_csv_path,
            pdf_name=entry.file_name if hasattr(entry, "file_name") else None,
            top_k=3
        )
        
        # Prepare the question
        question = entry.question
        if hasattr(entry, "question_type") and entry.question_type == "closed":
            question += " Return the answer in a single 'yes' or 'no' only. Do not provide additional explanations."
        
        # Create variables to capture the response object outside the lambda
        response_obj = None
        
        # Define a function that gets the response and captures the full response object
        def get_rag_response():
            nonlocal response_obj
            response_obj = chat_client.chat(question)
            return str(response_obj.response)
        
        # Use the response processor's QA response handling with retries
        cleaned_response = self.process_qa_response_with_retries(
            get_response_func=get_rag_response,
            question_type=getattr(entry, "question_type", ""),
            max_attempts=max_attempts
        )
        
        # If no valid response was generated, return an empty string
        if cleaned_response is None:
            cleaned_response = ""
            logger.warning(f"Failed to generate valid RAG response after {max_attempts} attempts")
            # Ensure response_obj is not None for source extraction
            if response_obj is None:
                response_obj = AgentChatResponse()
        
        # Extract source nodes and chunks from the response object
        source_nodes = []
        chunks = []
        
        if response_obj and hasattr(response_obj, 'source_nodes'):
            # Extract source nodes
            source_nodes = [node.metadata.get('doc_id', 'Unknown') for node in response_obj.source_nodes]
            # Extract chunks
            chunks = [node.text for node in response_obj.source_nodes]
        
        # Ensure we always return exactly 3 items for compatibility
        source_nodes = (source_nodes + [''] * 3)[:3]
        chunks = (chunks + [''] * 3)[:3]
        
        return cleaned_response, chunks, source_nodes
    
    def _process_direct_query(
            self, 
            entry: QABenchmarkEntry, 
            **kwargs
            ) -> str:
        """
        Process a query with direct LLM use (no RAG).
        
        Args:
            entry: The benchmark entry to evaluate.
        
        Keyword Arguments:
            context_type: Type of context ('none', 'pdf', 'gold').
            json_directory: Directory with JSON chunks.
            max_attempts: Max attempts to generate a valid response.
            retry_delay: Delay in seconds
            
        Returns:
            response (str): Response from LLM.
        """
        # Get the prompt with context if provided
        prompt,_ = self.prepare_prompt(entry, **kwargs)

        # Return None if no prompt is created
        if not prompt:
            return None

        # Use the response processor's QA response handling with retries
        max_attempts = kwargs.get('max_attempts', 3)
        retry_delay = kwargs.get('retry_delay', 10)
        cleaned_response = self.process_qa_response_with_retries(
            get_response_func=lambda: self.llm_handler.generate_response(prompt),
            question_type=getattr(entry, "question_type", ""),
            max_attempts=max_attempts,
            retry_delay=retry_delay
            )
        
        # If no valid response was generated, return an empty string
        if cleaned_response is None:
            logger.warning(f"Failed to generate valid response for entry {entry.entry_id} after {max_attempts} attempts")
            return None
        
        # No chunks or source nodes for direct queries
        return cleaned_response

    def evaluate_entry(
            self, 
            entry: QABenchmarkEntry,
            **kwargs
            ) -> Dict[str, Any]:
        """
        Evaluate a single QA benchmark entry with the specified context type.
        
        Args:
            entry: The benchmark entry to evaluate.
            
        Keyword Arguments:
            context_type: Type of context ('none', 'pdf', 'rag', 'gold').
            chromadb_path: Path to ChromaDB for RAG.
            collection_name: ChromaDB collection name.
            json_directory: Directory with JSON chunks.
            max_attempts: Max attempts to generate a valid response.
            retry_delay: Delay in seconds
            
        Returns:
            Dictionary with evaluation results.
        """
        context_type = kwargs.get('context_type', None)
        max_attempts = kwargs.get('max_attempts', 3)
        nepabench_directory = kwargs.get('nepabench_directory', None)
        
        # Validate context type and requirements
        if context_type:
            context_type = context_type.lower()
            self._validate_context_requirements(entry, context_type)
        
        # Process based on context type
        if not context_type:
            context_type = "none"
        
        if context_type in ["none", "pdf", "gold"]:
            response = self._process_direct_query(entry, **kwargs)
            chunks = [''] * 3
            source_nodes = [''] * 3
            
        elif context_type == "rag":
            logger.info("Context type: RAG - Using retrieval augmented generation")
            chromadb_path=kwargs.get('chromadb_path', None)
            doc_id_csv_path=kwargs.get('doc_id_csv_path', None)
            if nepabench_directory:
                chromadb_path = os.path.join(nepabench_directory, chromadb_path)
                doc_id_csv_path = os.path.join(nepabench_directory, doc_id_csv_path)
            response, chunks, source_nodes = self._process_rag_query(
                entry=entry,
                chromadb_path=chromadb_path,
                collection_name=kwargs.get('collection_name', None),
                doc_id_csv_path=doc_id_csv_path, 
                embed_model_name=kwargs.get('embed_model_name','all-MiniLM-L6-v2'),
                max_attempts=max_attempts
            )
            
        else:
            logger.error(f"Unsupported context type for evaluation: {context_type}")
            raise ValueError(f"Unsupported context type for evaluation: {context_type}")
        
        # Return None if response generation failed
        if response is None:
            return None
        
        # Create result dictionary
        result = {
            "question": entry.question,
            "answer_expected": entry.answer,
            "answer_predicted": response,
            "chunks": chunks,
            "source_nodes": source_nodes,
            "context_type": context_type
        }
        
        # Add optional fields if they exist
        for field in ['file_name', 'page_number', 'question_type', 'entry_id']:
            if hasattr(entry, field) and getattr(entry, field) is not None:
                result[field] = getattr(entry, field)
        
        logger.info(f"Question answering task completed for entry {entry.entry_id}")
        
        return result
    
    def extract_response(self, 
            entry: QABenchmarkEntry, 
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
        cleaned_response = self.clean_qa_response(response, entry.question_type)
        # Create result dictionary
        result = {
            "question": entry.question,
            "answer_expected": entry.answer,
            "answer_predicted": cleaned_response
        }
        
        # Add optional fields if they exist
        for field in ['file_name', 'page_number', 'question_type', 'entry_id']:
            if hasattr(entry, field) and getattr(entry, field) is not None:
                result[field] = getattr(entry, field)
        return result
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    conf = OmegaConf.load('configs/eval_mistral_config.yaml')
    
    from utils import load_benchmark_entries
    entries = load_benchmark_entries(conf)

    
    # Load the AWS Bedrock hosted model
    provider = conf["model"]["provider"]
    model_name = conf["model"]["name"]
    max_tokens = conf["model"]["max_tokens"] if "max_tokens" in conf["model"] else 20000
    from llm_handlers import AWSBedrockHandler
    llm_client = AWSBedrockHandler(
        model_name, token_limit=max_tokens
        )
    
    # QA evaluator agent check
    eval_config = conf["evaluation"]
    qaEval_client = QAEvaluator(llm_client, eval_config["prompt_file"])
    response = qaEval_client.evaluate_entry(
        entries[0], 
        **eval_config["eval_kwargs"]
        )
    
    logger.info(response)
