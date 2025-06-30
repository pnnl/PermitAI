import os
import sys
from typing import Dict

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
from llm_handlers.base_handler import BaseLLMHandler
from llama_index.llms.azure_openai import AzureOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from utils.logging_utils import LogManager
    
if __name__ == "__main__":
    LogManager.initialize("logs/test_azure_handler.log")
    

logger = LogManager.get_logger("azure_handler")

class AzureOpenAIHandler(BaseLLMHandler):
    """
    Handler for Azure OpenAI service.

    This class extends BaseLLMHandler to provide specific implementation
    for Azure OpenAI service.
    """
    def get_external_credentials(self)->Dict:
        """
        Retrieve Azure OpenAI credentials from environment variables.

        Returns:
            dict: A dictionary containing Azure OpenAI credentials.
        """
        return {
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "https://policyai-openai-westus.openai.azure.com/"),
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY", None),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
            "engine": os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4-turbo-preview")
        }

    def get_llm(self)->AzureOpenAI:
        """
        Get an Azure OpenAI LLM client.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI LLM client.
        """
        llm = AzureOpenAI(
            engine=self.client_kwargs["engine"],
            temperature=0.0,
            azure_endpoint=self.client_kwargs["azure_endpoint"],
            api_key=self.client_kwargs["api_key"],
            api_version=self.client_kwargs["api_version"],
        )
        logger.info(f"Azure OpenAI (model name: {self.model_name}) is successfully connected.")
        return llm
    
    def generate_response(self, prompt)->str:
        """
        Generate a response using the Azure OpenAI LLM.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """
        response = self.llm.complete(prompt)
        return response.text.strip()


class AzureChatOpenAIHandler(AzureOpenAIHandler):
    """
    Handler for Azure Chat OpenAI service.

    This class extends AzureOpenAIHandler to provide specific implementation
    for Azure Chat OpenAI service, which is compatible with RAGAS evaluation.
    """
    def get_llm(self)->AzureChatOpenAI:
        llm_model = AzureChatOpenAI(
            api_version=self.client_kwargs["api_version"],
            api_key=self.client_kwargs["api_key"],
            azure_endpoint=self.client_kwargs["azure_endpoint"],
            azure_deployment=self.client_kwargs["engine"],
            model='gpt-4',
            validate_base_url=False,
            )
        logger.info(f"Azure Chat OpenAI (model name: {self.model_name}) is successfully connected (compatible for RAGAS evaluation).")
        return llm_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    logger.info("Loading AZURE credentials as environment variables.")
    llm_client = AzureOpenAIHandler(model_name="gpt-4-turbo-preview")