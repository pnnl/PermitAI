import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers.base_handler import BaseLLMHandler
from google.oauth2 import service_account
from llama_index.llms.vertex import Vertex
from google.cloud.aiplatform_v1beta1.types import content
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_vertex_handler.log")

logger = LogManager.get_logger("vertex_handler")

class VertexGeminiHandler(BaseLLMHandler):
    """
    Handler for Google Vertex AI Gemini model.

    This class extends BaseLLMHandler to provide specific implementation
    for Google Vertex AI Gemini model.
    """
    
    def get_external_credentials(self, credentials_json_path=None):
        """
        Retrieve Google Cloud credentials from a service account JSON file.

        Args:
            credentials_json_path (str, optional): Path to the service account JSON file. 
                If None, it will be retrieved from the VERTEXAI_CREDENTIALS_JSON_PATH environment variable.

        Returns:
            google.oauth2.service_account.Credentials: The Google Cloud credentials.

        Raises:
            ValueError: If the credentials file is not found or is malformed.
        """
        if credentials_json_path is None:
            credentials_json_path=os.environ.get("VERTEXAI_CREDENTIALS_JSON_PATH", None)

        try:
            credentials: service_account.Credentials = (service_account.Credentials.from_service_account_file(credentials_json_path))
            logger.info(f"Accessing credentials for Google Gemini from {credentials_json_path}")
        except:
            raise ValueError("MalformedError: Please check your credentials !!")
            
        return credentials

    def get_llm(self)->Vertex:
        """
        Get a Google Vertex AI LLM client.

        Returns:
            Vertex: An instance of the Vertex LLM client.
        """
        # Safety settings
        safety_settings = {
            content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT : content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HARASSMENT : content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HATE_SPEECH : content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT : content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_UNSPECIFIED : content.SafetySetting.HarmBlockThreshold.BLOCK_NONE
        }

        # Vertex model
        llm = Vertex(
            model = self.model_name, 
            project = self.client_kwargs.project_id, 
            credentials = self.client_kwargs,
            safety_settings = safety_settings
        )
        logger.info(f"Google Vertex AI (model name: {self.model_name}) is successfully connected.")
        return llm
    
    def generate_response(self, prompt):
        """
        Generate a response using the Google Vertex AI LLM.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """
        response = self.llm.complete(prompt)
        return response.text.strip()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    logger.info("Loading credential JSON file path as environment variable.")
    llm_client = VertexGeminiHandler(model_name="gemini-1.5-pro-preview-0409")
    system_prompt = "You are an assistant who will answer a query."
    user_prompt = "What is the capital of India?"
    context_prompt = "\nUse only the following context to answer the query: The capital of India before 1947 was Calcutta."
    response = llm_client.generate_response(system_prompt+user_prompt+context_prompt)
    logger.info(f"Query: {user_prompt}")
    logger.info(f"Response: {response}")