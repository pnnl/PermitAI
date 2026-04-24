from abc import ABC, abstractmethod
from typing import Dict

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
    """
    def __init__(self, model_name):
        """
        Initialize the BaseLLMHandler.

        Args:
            model_name (str): The name of the LLM model to use.
        """
        self.model_name = model_name
        self.client_kwargs = self.get_external_credentials()
        self.llm = self.get_llm()

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
    def generate_response(self, user_prompt)->str:
        """
        Generate a response using the LLM.

        Args:
            user_prompt (str): The user prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """
        pass