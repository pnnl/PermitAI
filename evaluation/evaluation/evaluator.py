import os
import sys
import pandas as pd
import chromadb
import tiktoken
import re
import json
from typing import List, Tuple, Optional
import warnings

from llama_index.core.chat_engine.types import BaseChatEngine, AgentChatResponse

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers.base_handler import BaseLLMHandler
from utils.dataloader import extract_unique_ids, BenchmarkEntry

# Import necessary as ServiceContext is deprecated 
from llama_index.core import Settings

# Import necessary to avoid Assertion Error while loading the Embedding Model
# Only accepted formats are Langchain embeddings and HuggingFaceEmbedding starting with 'local'
from langchain_huggingface import HuggingFaceEmbeddings


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import ExactMatchFilter

import traceback
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_evaluator.log")

logger = LogManager.get_logger(__name__)

def count_tokens(text):
    """
    Count the number of tokens in a given text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def truncate_text(text, max_tokens):
    """
    Truncate the input text to a specified maximum number of tokens.

    Args:
        text (str): The input text to truncate.
        max_tokens (int): The maximum number of tokens to keep.

    Returns:
        str: The truncated text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

class Evaluator:
    """
    A class for evaluating language models using various context types and benchmarks.
    """
    def __init__(self, llm_handler: BaseLLMHandler, prompt_file: str):
        """
        Initialize the Evaluator.

        Args:
            llm_handler (BaseLLMHandler): The language model handler.
            prompt_file (str): Path to the file containing the prompt template.
        """
        self.llm_handler = llm_handler
        self.prompt_file = prompt_file
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
    
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Read the prompt file and format it with the given question and context.

        Args:
            question (str): The question to be answered.
            context (Optional[str]): The context for the question.

        Returns:
            str: The formatted prompt.

        Raises:
            ValueError: If context is None and the prompt requires a context.
        """
        try:
            with open(self.prompt_file, 'r') as file:
                prompt_template = file.read()

            if '{context}' in prompt_template and context is None:
                raise ValueError("The prompt requires a context. Either provide a context or change the prompt file to one with no placeholder for context.")

            if '{context}' not in prompt_template and context is not None:
                warn_msg = "Context provided but prompt file has no {context} placeholder. The context will not be used in the prompt."
                logger.warning(warn_msg)
                warnings.warn(warn_msg, UserWarning)

            formatted_prompt = prompt_template.replace('{question}', question)
            if context and '{context}' in prompt_template:
                formatted_prompt = formatted_prompt.replace('{context}', context)

            logger.info(f"Prompt formatted successfully for question: {question[:50]}...")
            return formatted_prompt

        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise

    def load_chromadb(self, chromadb_path: str, collection_name: str, embed_model: str = 'BAAI/bge-small-en-v1.5'):
        """
        Load ChromaDB collection for RAG.

        Args:
            chromadb_path (str): Path to the ChromaDB database.
            collection_name (str): Name of the collection to load.
            embed_model (str, optional): Name of the embedding model to use. Default is BAAI/bge-small-en-v1.5

        Returns:
            VectorStoreIndex: The loaded index.

        Raises:
            FileNotFoundError: If the ChromaDB database file cannot be found.
        """
        if not os.path.exists(chromadb_path):
            logger.debug("Incorrect path to the Chromadb database")
            raise FileNotFoundError("The chromadb database file cannot be found.")
        
        db = chromadb.PersistentClient(path=chromadb_path)
        # Update global settings with HuggingFaceEmbeddings instance
        Settings.embed_model = HuggingFaceEmbeddings(model_name=embed_model)
        
        chroma_collection = db.get_or_create_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context = storage_context
        )
        logger.info(f"Chroma Collection {collection_name} loaded ---- Chroma count: {chroma_collection.count()}")
        return index
    
    def should_retry(self, response: str, benchmark: BenchmarkEntry) -> bool:
        """
        Determine if the response should be retried based on certain conditions.
        
        Args:
            response (str): The generated response.
            benchmark (BenchmarkEntry): The benchmark entry being evaluated.
        
        Returns:
            bool: True if the response should be retried, False otherwise.
        """
        if response == "Empty Response":
            return True
        if response.startswith("I'm sorry, but the context"):
            return True
        if benchmark.question_type == "closed" and not (response.lower().startswith("yes") or response.lower().startswith("no")):
            logger.warning("Closed type question : Not answered in correct format")
            return True
        return False
    
    def clean_response(self, response: str, benchmark: BenchmarkEntry) -> str:
        """
        Clean the response by removing system tokens and unnecessary whitespace.
        For closed questions, return only "yes" or "no".
        
        Args:
            response (str): The raw response from the language model.
            benchmark (BenchmarkEntry): The benchmark entry being evaluated.
        
        Returns:
            str: The cleaned response.
        """
        # Remove HTML-like decorators
        cleaned = re.sub(r'<<[^>]*>>', '', response)  # Remove opening tags
        cleaned = re.sub(r'<</[^>]*>>', '', cleaned)  # Remove closing tags
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if benchmark.question_type == 'closed':
            # For closed questions, return only "yes" or "no"
            lower_cleaned = cleaned.lower()
            if 'not sure' in lower_cleaned:
                return "unclear"
            elif 'yes' in lower_cleaned or 'true' in lower_cleaned or 'correct' in lower_cleaned:
                return "yes"
            elif 'no' in lower_cleaned or 'false' in lower_cleaned or 'incorrect' in lower_cleaned:
                return "no"
            else:
                logger.warning(f"Unclear response for closed question: {cleaned}")
                return "unclear"
        
        return cleaned
    
    def setup_rag_chat_client(
            self, chromadb_path: str,collection_name: str,
            docID_csvpath: str = None,
            pdf_name: str = None, top_k: int = 3, 
            embed_model: str = 'BAAI/bge-small-en-v1.5'
            ) -> BaseChatEngine:
        """
        Create a chat client with the required setting for RAG.

        Args:
            chromadb_path (str): Path to the ChromaDB database.
            collection_name (str): Name of the collection to use.
            docID_csvpath (str, optional): Path to the CSV file containing document IDs.
            pdf_name (str, optional): Name of the PDF to filter by.
            top_k (int, optional): Number of top results to consider.
            embed_model (str, optional): Name of the embedding model to use. Default is BAAI/bge-small-en-v1.5

        Returns:
            BaseChatEngine: The configured chat engine.

        Raises:
            ValueError: If required parameters are missing.
        """
        if not chromadb_path:
            logger.error("Chromadb path must be provided when using RAG setup.")
            raise ValueError("Need to provide a path to the Chromadb database.")
        
        if not collection_name:
            logger.error("Chromadb collection name must be provided when using RAG setup.")
            raise ValueError("Need to provide a collection name to load documents from the Chromadb database.")
        
        index = self.load_chromadb(chromadb_path=chromadb_path, collection_name=collection_name, embed_model=embed_model)
        if pdf_name:
            if not docID_csvpath:
                logger.error("A document to Chromadb ID mapping CSV must be provided when filtering based on PDF name.")
                raise ValueError("Need to provide a CSV file with document to Chromadb metadata ID.")
            pdf_to_uniqueID = extract_unique_ids(docID_csvpath)
            select_doc_id = pdf_to_uniqueID[pdf_name]
            if not select_doc_id:
                raise ValueError(f"No unique_id found for PDF: {pdf_name}")
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="unique_id", value=select_doc_id)] 
                )
            logger.info(f"Setting up chat engine with only content from {pdf_name} -- doc ID {select_doc_id}")
        else:
            logger.info(f"Setting up chat engine with all documents as context")
            filters = MetadataFilters(filters=[])

            
        # initialize the chat engine with context with/without filter on the pdf
        chat_engine = index.as_chat_engine(
            chat_mode="context",  # NOTE: Changed from 'condensed_question' to 'context' by Rounak based on the documentation
            similarity_top_k=top_k, 
            response_mode="tree_summarize", 
            llm=self.llm_handler.llm, 
            filters=filters
            )
        return chat_engine
    
    def use_rag_chat_client(
            self, client: BaseChatEngine, 
            benchmark: BenchmarkEntry, 
            max_attempts: int = 3
            ) -> Tuple[str, List[str], List[str]]:
        """
        Respond to the prompt using the chat engine with multiple attempts.

        Args:
            client (BaseChatEngine): The chat client that has been setup with the chromadb index.
            benchmark (BenchmarkEntry): The benchmark question entry that is being asked.
            max_attempts (int, optional): Maximum number of attempts to generate a response.

        Returns:
            Tuple[str, List[str], List[str]]: The cleaned response, chunks, and source nodes.
        """
        # additional prompt for 'closed' type of question
        question = benchmark.question
        if benchmark.question_type == "closed":
            question += " Return the answer in a single 'yes' or 'no' only. Do not provide additional explanations."
        
        # multiple attempts to answer prompt
        index_attempt_question = 0
        while(index_attempt_question < max_attempts):
            try:
                index_attempt_question += 1
                logger.info(f"Attempt {index_attempt_question} to respond to question.")
                
                response = client.chat(question)
                raw_response = str(response.response)
                # Clean the response
                cleaned_response = self.clean_response(raw_response, benchmark)

                if not self.should_retry(cleaned_response, benchmark):
                    break
            
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                cleaned_response = ''
                response = AgentChatResponse()
        
        # Extract source nodes and ensure we have exactly 3 
        # NOTE: Currently we only save 3 source node information in the CSV file
        source_nodes = [node.metadata.get('doc_id', 'Unknown') for node in response.source_nodes]
        source_nodes = (source_nodes + [''] * 3)[:3]
        
        # Ensure we always have exactly 3 chunks
        # NOTE: Currently we only save 3 chunks in the CSV file
        chunks = [node.text for node in response.source_nodes]
        chunks = (chunks + [''] * 3)[:3]
        
        return cleaned_response, chunks, source_nodes
    
    def use_llm_client(self, benchmark: BenchmarkEntry, context: str = None, max_attempts: int = 3):
        """
        Respond to the prompt using the LLM handler with multiple attempts.

        Args:
            benchmark (BenchmarkEntry): The benchmark entry being evaluated.
            context (str, optional): Additional context for the question.
            max_attempts (int, optional): Maximum number of attempts to generate a response.

        Returns:
            Tuple[str, List[str], List[str]]: The cleaned response, chunks, and source nodes.
        """
        # additional prompt for 'closed' type of question
        question = benchmark.question
        if benchmark.question_type == "closed":
            question += " Return the answer in a single 'yes' or 'no' only. Do not provide additional explanations."
        
        # Provide long context within prompt if flag is enabled and generate formatted prompt
        prompt = self.format_prompt(question, context)
        
        # multiple attempts to answer prompt
        index_attempt_question = 0
        while(index_attempt_question < max_attempts):
            try:
                index_attempt_question += 1
                logger.info(f"Attempt {index_attempt_question} to respond to question.")
                
                raw_response = self.llm_handler.generate_response(prompt)
                # Clean the response
                cleaned_response = self.clean_response(raw_response, benchmark)

                if not self.should_retry(cleaned_response, benchmark):
                    break
            
            except Exception as e:
                logger.error(f"Error while responding to prompt: {e}")
                cleaned_response = ""
        
        # empty strings for source nodes and chunks
        source_nodes = [''] * 3
        chunks = [''] * 3
        
        return cleaned_response, chunks, source_nodes
    
    def get_pdf_context(self, json_dir: str, pdf_name: str, max_tokens: int = 100000) -> str:
        """
        Get the entire PDF content from the extracted information in the JSON files.

        Args:
            json_dir (str): Directory containing the JSON files with PDF chunks.
            pdf_name (str): Name of the PDF file.
            max_tokens (int, optional): Maximum number of tokens to include in the context.

        Returns:
            str: The PDF context.

        Raises:
            ValueError: If required parameters are missing.
            FileNotFoundError: If the JSON file for the PDF is not found.
        """
        if not json_dir:
            logger.error("JSON directory path must be provided when using PDF context.")
            raise ValueError("Need to provide a path to the JSON directory with PDF chunks.")
        
        if not pdf_name:
            logger.error("PDF name must be provided when using PDF context.")
            raise ValueError("Need to provide the name of the PDF file.")
        elif pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
            pdf_name = pdf_name.replace(" ","_")

        # Append "_output.json" to the file_name
        json_file = f"{json_dir}{pdf_name}_output.json"
        if not os.path.exists(json_file):
            logger.error(f"JSON file {json_file} does not exist")
            raise FileNotFoundError(f"JSON file {json_file} with PDF chunks not found")

        # Extract text from the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

            pdf_context = ""
            for page_key in data:
                pdf_context += data[page_key]['text']
        
        # Truncate the input text if it exceeds the max_tokens limit
        token_count = count_tokens(pdf_context)
        logger.info(f"Current token count = {token_count}")
        if token_count > max_tokens:
            logger.info(f"Truncating the context because it exceeds maximum token count {max_tokens}")
            pdf_context = truncate_text(pdf_context, max_tokens)
        return pdf_context

    def generate_response(
            self, benchmark: BenchmarkEntry, 
            context_type: str = None,
            chromadb_path: str = None,
            collection_name: str = None,
            json_directory: str = None,
            docID_csvpath: str = None,
            max_attempts: int = 3, 
            embed_model: str = 'BAAI/bge-small-en-v1.5',
            max_tokens: int = 100000
            ) -> Tuple[str, List[str], List[str]]:
        """
        Generate a response for a given question using the specified context type.

        Args:
            benchmark (BenchmarkEntry): The benchmark entry being evaluated.
            context_type (str, optional): Type of context to use ('none', 'pdf', 'rag', or 'gold').
            chromadb_path (str, optional): Path to the ChromaDB database.
            collection_name (str, optional): Name of the ChromaDB collection.
            json_directory (str, optional): Directory containing JSON files with PDF chunks.
            docID_csvpath (str, optional): Path to the CSV file containing document IDs.
            max_attempts (int, optional): Maximum number of attempts to generate a response.
            embed_model (str, optional): Name of the embedding model to use. Default is BAAI/bge-small-en-v1.5
            max_tokens (int, optional): Maximum number of tokens for PDF context.

        Returns:
            Tuple[str, List[str], List[str]]: The response, chunks, and source nodes.

        Raises:
            ValueError: If required attributes are missing for the specified context type.
            NotImplementedError: If an unknown context type is specified.
        """

        # Validate context type and required attributes
        if context_type:
            context_type = context_type.lower()
            
            # Check for PDF context requirements
            if context_type == "pdf" and not benchmark.file_name:
                error_msg = ("Context type 'pdf' requires benchmark entry to have 'file_name' attribute. "
                            "This is required to locate the corresponding PDF file.")
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Check for RAG context requirements
            elif context_type == "rag" and not benchmark.file_name:
                error_msg = ("Context type 'rag' requires benchmark entry to have 'file_name' attribute. "
                            "This is required to filter relevant documents in the RAG setup.")
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Check for Gold context requirements
            elif context_type == "gold" and not benchmark.context:
                error_msg = ("Context type 'gold' requires benchmark entry to have 'context' attribute. "
                            "This is required as the ground truth context for response generation.")
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # call methods based on type of context
        if not context_type or context_type.lower() == "none":
            logger.info("Context type: NONE --> Direct query-response with no context.")
            response, chunks, source_nodes = self.use_llm_client(
                benchmark=benchmark, 
                context=None, 
                max_attempts=max_attempts
                )

        elif context_type.lower() == "pdf":
            logger.info("Context type: PDF --> Direct query-response with entire PDF content")
            pdf_context = self.get_pdf_context(
                json_dir=json_directory, 
                pdf_name=benchmark.file_name, 
                max_tokens=max_tokens
                )
            response, chunks, source_nodes = self.use_llm_client(
                benchmark=benchmark, 
                context=pdf_context, 
                max_attempts=max_attempts
                )
        
        elif context_type.lower() == "rag":
            logger.info("Context type: RAG --> Regular RAG setup with filter on the PDF")
            chat_client = self.setup_rag_chat_client(
                chromadb_path=chromadb_path, collection_name=collection_name,
                docID_csvpath=docID_csvpath,
                pdf_name=benchmark.file_name, # NOTE: We do a filter on PDF to be consistent with scripts written by Hung
                top_k=3, embed_model=embed_model
                )
            response, chunks, source_nodes = self.use_rag_chat_client(
                client=chat_client, 
                benchmark=benchmark, 
                max_attempts=max_attempts
                )
        
        elif context_type.lower() == "gold":
            logger.info("Context type: Gold --> Long context propting with prompt containing context from which answer can be found")
            response, chunks, source_nodes = self.use_llm_client(
                benchmark=benchmark, 
                context=benchmark.context, 
                max_attempts=max_attempts
                )
        
        else:
            logger.error(f"Unknown context type: {context_type}. Choices are ['none', 'pdf', 'rag', 'gold']")
            raise NotImplementedError("Response generation not implemented for requested context type.")

        return response, chunks, source_nodes

    def evaluate_benchmark(
            self, benchmark_data: List[BenchmarkEntry], output_path: str, 
            context_type: str = None, 
            chromadb_path: str = None, collection_name: str = None,
            json_directory: str = None, 
            docID_csvpath: str = None,
            max_attempts: int = 3, 
            embed_model: str = 'BAAI/bge-small-en-v1.5',
            continue_from_previous: bool = True, 
            max_tokens: int = 100000
            ):
        """
        Evaluate the benchmark questions and update the CSV file with results.

        Args:
            benchmark_data (List[BenchmarkEntry]): List of benchmark entries to evaluate.
            output_path (str): Path to save the results CSV file.
            context_type (str, optional): Type of context to use.
            chromadb_path (str, optional): Path to the ChromaDB database.
            collection_name (str, optional): Name of the ChromaDB collection.
            json_directory (str, optional): Directory containing JSON files with PDF chunks.
            docID_csvpath (str, optional): Path to the CSV file containing document IDs.
            max_attempts (int, optional): Maximum number of attempts to generate a response.
            embed_model (str, optional): Name of the embedding model to use. Default is BAAI/bge-small-en-v1.5
            continue_from_previous (bool, optional): Whether to continue from a previous evaluation.
            max_tokens (int, optional): Maximum number of tokens for PDF context.
        """

        # Define standard column order
        standard_columns = [
            'file_name', 'page_number', 'question_type', 'question', 
            'answer_expected', 'answer_predicted',
            'chunk_1', 'chunk_2', 'chunk_3',
            'source_1', 'source_2', 'source_3'
        ]

        # Check if the output CSV file exists
        if continue_from_previous:
            if not os.path.exists(output_path):
                logger.info(f"No file {output_path} found with previous responses")
                results_df = pd.DataFrame()
            else:
                results_df = pd.read_csv(output_path)
                logger.info(f"CSV file {output_path} exists with {len(results_df)}/{len(benchmark_data)} recorded responses.")
                start_index = len(results_df)
                benchmark_data = benchmark_data[start_index:]
                logger.info(f"Generating response for remaining {len(benchmark_data)} questions.")
        else:
            logger.info("Starting response generation for all questions.")
            results_df = pd.DataFrame()
        
        # Process benchmark entries
        for i,entry in enumerate(benchmark_data):
            logger.info(f"Handling benchmark question {i+1} out of {len(benchmark_data)}")
            response, chunks, source_nodes = self.generate_response(
                entry, context_type=context_type, 
                chromadb_path=chromadb_path, collection_name=collection_name, 
                json_directory=json_directory,
                docID_csvpath=docID_csvpath,
                max_attempts=max_attempts, 
                embed_model=embed_model,
                max_tokens=max_tokens
                )
            
            # Initialize result dictionary with None values for all standard columns
            result = {col: None for col in standard_columns}
            
            # Update with available values
            result.update({
                'question': entry.question,
                'answer_expected': entry.answer,
                'answer_predicted': response
            })

            # Add optional fields if they exist
            if entry.file_name is not None:
                result['file_name'] = entry.file_name
            if entry.page_number is not None:
                result['page_number'] = entry.page_number
            if entry.question_type is not None:
                result['question_type'] = entry.question_type

            # Add chunks if available
            for j, chunk in enumerate(chunks[:3], 1):
                result[f'chunk_{i}'] = chunk

            # Add source nodes if available
            for j, source in enumerate(source_nodes[:3], 1):
                result[f'source_{i}'] = source

            # Create a single-row DataFrame with the result
            result_df = pd.DataFrame([result])

            # Ensure columns are in the standard order
            # Only include columns that have data (not all None)
            available_columns = [col for col in standard_columns 
                               if col in result_df.columns and 
                               not result_df[col].isna().all()]
            result_df = result_df[available_columns]

            # If results_df is empty, use the new result_df
            if results_df.empty:
                results_df = result_df
            else:
                # Get the union of columns in correct order
                all_columns = [col for col in standard_columns 
                             if col in results_df.columns or col in result_df.columns]
                
                # Add missing columns with None values
                for col in all_columns:
                    if col not in results_df.columns:
                        results_df[col] = None
                    if col not in result_df.columns:
                        result_df[col] = None

                # Append the result and reorder columns
                results_df = pd.concat([results_df, result_df], ignore_index=True)
                results_df = results_df[all_columns]

            # Write the updated DataFrame to CSV
            results_df.to_csv(output_path, index=False)
            logger.info(f"Entry {i+1} processed and saved to CSV path {output_path}.")
