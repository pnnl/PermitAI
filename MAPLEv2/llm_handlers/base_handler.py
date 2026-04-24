from abc import ABC, abstractmethod
from typing import Dict, Optional, Type
from pydantic import BaseModel

class BaseLLMHandler(ABC):
    """
    Abstract base class for LLM handlers.

    This class defines the interface for LLM handlers. Each concrete implementation
    should have an attribute `model_name` and implement the following methods:
    - get_external_credentials()
    - get_llm()
    - generate_response(user_prompt)

    Attributes:
        model_name (str): The name of the LLM model.
        client_kwargs (dict): A dictionary containing the credentials for the API.
        llm: An LLM client for chat completion tasks.
        rag_llm: An LLM client for RAG tasks.
    """
    def __init__(self, model_name, token_limit=100000):
        """
        Initialize the BaseLLMHandler.

        Args:
            model_name (str): The name of the LLM model to use.
            token_limit (int): The maximum number of token for the prompt
        """
        self.model_name = model_name
        self.token_limit = token_limit
        self.client_kwargs = self.get_external_credentials()
        self.llm = self.get_llm()
        self.rag_llm = self.get_rag_llm()

    @abstractmethod
    def get_external_credentials(self)->Dict:
        """
        Retrieve external credentials for the API.

        Returns:
            dict: A dictionary containing the credentials for the API.
        """
        pass

    @abstractmethod
    def get_llm(self):
        """
        Get an LLM client for chat completion tasks.

        Returns:
            object: An LLM client object.
        """
        pass

    @abstractmethod
    def get_rag_llm(self):
        """
        Get an LLM RAG client for chat completion tasks.

        Returns:
            object: An Llamaindex LLM client object.
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, response_format: Optional[Type[BaseModel]] = None)->str:
        """
        Generate a response using the LLM.

        Args:
            prompt (str): The user prompt to generate a response for.
            response_format (Optional[Type[BaseModel]]): Pydantic model for structured output.

        Returns:
            str: The generated response from the LLM.
        """
        pass