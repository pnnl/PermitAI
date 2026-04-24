import os, sys

try:
    import chromadb
except Exception as e:
    print(f"Error loading chromadb python module: {e}")

from typing import Dict, Optional
import pandas as pd

# Import Settings necessary as ServiceContext is deprecated 
# Import necessary to avoid Assertion Error while loading the Embedding Model
# Only accepted formats are Langchain embeddings and HuggingFaceEmbedding starting with 'local'

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.vector_stores.types import ExactMatchFilter
from llama_index.core.chat_engine.types import BaseChatEngine
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager
from llm_handlers import BaseLLMHandler

logger = LogManager.get_logger(__name__)

def extract_unique_ids(csv_path: str) -> Dict[str, str]:
    """
    Extract unique_id for each pdf_name from the given CSV file.
    Remove double quotes from unique_id values.
    
    Args:
        csv_path (str): Path to the CSV file containing the mapping.
    
    Returns:
        Dict[str, str]: A dictionary mapping pdf_name to unique_id.
        Returns an empty dictionary if the file is empty or doesn't contain required columns.
    """
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return {}
            
        df = pd.read_csv(csv_path)
        
        # Check if the DataFrame is empty or missing required columns
        if df.empty or not all(col in df.columns for col in ['pdf_name', 'unique_id']):
            logger.warning(f"CSV file {csv_path} is empty or missing required columns ['pdf_name', 'unique_id']")
            return {}
        
        # Remove double quotes from unique_id
        df['unique_id'] = df['unique_id'].str.replace('"', '')
        
        # Create the mapping dictionary
        mapping = dict(zip(df['pdf_name'], df['unique_id']))
        logger.info(f"Extracted {len(mapping)} pdf_name to unique_id mappings from {csv_path}")
        return mapping
        
    except Exception as e:
        logger.error(f"Error extracting unique IDs from {csv_path}: {str(e)}")
        return {}

def load_chromadb(chromadb_path: str, 
                 collection_name: str, 
                 embed_model_name: str) -> VectorStoreIndex:
    """
    Load ChromaDB collection for RAG.
    
    Args:
        chromadb_path (str): Path to the ChromaDB database.
        collection_name (str): Name of the collection to load.
        embed_model_name (str): Name of the embedding model to use.
        
    Returns:
        VectorStoreIndex: The loaded vector index.
        
    Raises:
        FileNotFoundError: If the ChromaDB database cannot be found.
    """
    if not os.path.exists(chromadb_path):
        logger.error(f"ChromaDB path does not exist: {chromadb_path}")
        raise FileNotFoundError(f"The chromadb database file cannot be found at {chromadb_path}")
    
    try:
        # Create persistent client
        db = chromadb.PersistentClient(path=chromadb_path)
        
        # Update global settings with embedding model
        Settings.embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
        
        # Get or create collection
        chroma_collection = db.get_or_create_collection(collection_name)
        
        # Create vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
        
        logger.info(f"Successfully loaded Chroma Collection '{collection_name}' with {chroma_collection.count()} documents")
        return index
        
    except Exception as e:
        logger.error(f"Error loading ChromaDB: {str(e)}")
        raise

def setup_rag_chat_client(
        chromadb_path: str, 
        collection_name: str,
        embed_model_name: str,
        llm_handler: BaseLLMHandler,
        id_mapping: Optional[Dict[str, str]] = None,
        doc_id_csv_path: Optional[str] = None,
        pdf_name: Optional[str] = None,
        top_k: int = 3
        ) -> BaseChatEngine:
    """
    Set up RAG chat client with optional document filtering.
    
    Args:
        chromadb_path (str): Path to ChromaDB.
        collection_name (str): ChromaDB collection name.
        embed_model_name (str): Name of the embedding model to use.
        llm_handler (BaseLLMHandler): The language model to use for generating responses.
        id_mapping (Optional[Dict[str, str]]): Mapping from PDF names to unique IDs.
        doc_id_csv_path (Optional[str]): Path to CSV with document ID mappings (used if pdf_to_uniqueID is None).
        pdf_name (Optional[str]): Optional name of PDF to filter by.
        top_k (int): Number of top results to consider. Defaults to 3.
        
    Returns:
        BaseChatEngine: The configured chat engine.
        
    Raises:
        ValueError: If embedding model is not provided.
    """
    if not embed_model_name:
        raise ValueError("Embedding model name must be provided for RAG setup")
        
    # Load ChromaDB
    index = load_chromadb(
        chromadb_path=chromadb_path, 
        collection_name=collection_name,
        embed_model_name=embed_model_name
    )
    
    # Load document ID mapping if not provided but CSV path is available
    if id_mapping is None and doc_id_csv_path:
        id_mapping = extract_unique_ids(doc_id_csv_path)
        logger.info(f"Loaded document ID mapping from {doc_id_csv_path}")
    
    # Apply document filter if pdf_name is provided and mapping is available
    if pdf_name and id_mapping and pdf_name in id_mapping:
        select_doc_id = id_mapping[pdf_name]
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unique_id", value=select_doc_id)]
        )
        logger.info(f"Setting up chat engine filtered for document: {pdf_name} (ID: {select_doc_id})")
    elif pdf_name and id_mapping and f"{pdf_name}.pdf" in id_mapping:
        select_doc_id = id_mapping[f"{pdf_name}.pdf"]
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="unique_id", value=select_doc_id)]
        )
        logger.info(f"Setting up chat engine filtered for document: {pdf_name} (ID: {select_doc_id})")
    else:
        if pdf_name and (id_mapping is None or pdf_name not in id_mapping):
            warning_msg = "Document ID mapping not available" if id_mapping is None else f"Document {pdf_name} not found in mapping"
            logger.warning(f"{warning_msg}. Using all documents for retrieval.")
        filters = MetadataFilters(filters=[])
        logger.info("Setting up chat engine with all documents (no filter)")
        
    # Initialize the chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=top_k, 
        response_mode="tree_summarize", 
        llm=llm_handler.rag_llm, 
        filters=filters
    )
    
    return chat_engine