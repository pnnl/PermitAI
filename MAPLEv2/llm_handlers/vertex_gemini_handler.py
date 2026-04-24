import os
import sys
from typing import Type, Optional, Dict, List, Union, Any

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers.base_handler import BaseLLMHandler
from google.oauth2 import service_account
from llama_index.llms.vertex import Vertex
from google.cloud.aiplatform_v1beta1.types import content

import vertexai
import json
from vertexai.generative_models import GenerativeModel, GenerationConfig, GenerationResponse, Part, Image
from pydantic import BaseModel
from utils.schema_utils import pydantic_to_vertex_schema

from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_vertex_handler.log")

logger = LogManager.get_logger("vertex_handler")


class LlamaVertexHandler(BaseLLMHandler):
    """
    Handler for Google Vertex AI Gemini model using LlamaIndex.
    
    This class provides basic text generation capabilities without structured output support.
    For structured output, use VertexAIHandler or GoogleGenAIHandler.
    """
    
    def get_external_credentials(self):
        """
        Retrieve Google Cloud credentials from a service account JSON file.

        Returns:
            google.oauth2.service_account.Credentials: The Google Cloud credentials.

        Raises:
            ValueError: If the credentials file is not found or is malformed.
        """
        credentials_json_path = os.environ.get("VERTEXAI_CREDENTIALS_JSON_PATH", None)
        try:
            credentials: service_account.Credentials = (
                service_account.Credentials.from_service_account_file(credentials_json_path)
            )
            
            # Extract project ID from credentials
            with open(credentials_json_path, 'r') as f:
                cred_data = json.load(f)
            
            logger.info(f"Accessing credentials for Google Gemini from {credentials_json_path}")
            logger.info(f"Project ID: {cred_data.get('project_id')}, Location: {os.environ.get('VERTEXAI_LOCATION', 'us-central1')}")
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            raise ValueError(f"MalformedError: Please check your credentials loaded from {credentials_json_path}!!")

        return credentials
        
    def get_llm(self) -> Vertex:
        """
        Get a Google Vertex AI LLM client using LlamaIndex.

        Returns:
            Vertex: An instance of the Vertex LLM client.
        """
        safety_settings = {
            content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_UNSPECIFIED: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE
        }
        
        # Vertex model using LlamaIndex
        llm = Vertex(
            model=self.model_name, 
            project=self.client_kwargs.project_id, 
            credentials=self.client_kwargs,
            safety_settings=safety_settings
        )
        logger.info(f"Google Vertex AI LlamaIndex client is successfully connected.")
        return llm
    
    def get_rag_llm(self) -> Vertex:
        """
        Get a Google Vertex AI RAG LLM client using LlamaIndex.

        Returns:
            Vertex: An instance of the Llamaindex Vertex LLM client.
        """
        safety_settings = {
            content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_UNSPECIFIED: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE
        }
        
        # Vertex model using LlamaIndex
        llm = Vertex(
            model=self.model_name, 
            project=self.client_kwargs.project_id, 
            credentials=self.client_kwargs,
            safety_settings=safety_settings
        )
        logger.info(f"Google Vertex AI LlamaIndex client is successfully connected for RAG operations.")
        return llm
    
    def generate_response(self, prompt: str, response_format: Optional[Type[BaseModel]] = None, images: Union[str, List[str]] = None) -> str:
        """
        Generate a response using the Google Vertex AI LLM via LlamaIndex.

        Args:
            prompt (str): The user prompt to generate a response for.
            response_format (Optional[Type[BaseModel]]): Pydantic model for structured output.
                If provided, an error will be logged as this handler doesn't support structured output.
            images (Optional[Union[str, List[str]]]): Image files to attach to prompt
                If provided, an error will be logged as this handler doesn't support images in prompt.

        Returns:
            str: The generated response from the LLM.
        """
        if response_format is not None:
            logger.error(
                "LlamaVertexHandler cannot generate output in a Pydantic format. "
                "Please use VertexAIHandler or GoogleGenAIHandler for structured output."
            )
            raise ValueError(
                "LlamaVertexHandler does not support structured output. "
                "Use VertexAIHandler or GoogleGenAIHandler instead."
            )
        
        if images is not None:
            logger.error(
                "LlamaVertexHandler cannot support images in prompt. "
                "Please use VertexAIHandler or GoogleGenAIHandler for images in prompt."
            )
            raise ValueError(
                "LlamaVertexHandler does not support images in prompt. "
                "Use VertexAIHandler or GoogleGenAIHandler instead."
            )
        
        # Use LlamaIndex for text generation
        response = self.llm.complete(prompt)
        return response.text.strip()


class VertexAIHandler(LlamaVertexHandler):
    """
    Handler for Google Vertex AI Gemini model with native Vertex AI SDK.
    
    This class provides both regular text generation and structured output capabilities
    using the native Vertex AI GenerativeModel.
    """
    
    def get_llm(self) -> GenerativeModel:
        """
        Get the native Vertex AI GenerativeModel for structured output.
        
        Returns:
            vertexai.generative_models.GenerativeModel: Native Vertex AI model.
        """
        # Configure safety settings
        safety_settings = {
            content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            content.HarmCategory.HARM_CATEGORY_UNSPECIFIED: content.SafetySetting.HarmBlockThreshold.BLOCK_NONE
        }

        # Initialize Vertex AI
        try:
            vertexai.init(
                project=self.client_kwargs.project_id,
                location=os.environ.get("VERTEXAI_LOCATION", "us-central1"),
                credentials=self.client_kwargs
            )
            logger.info(f"Initialized native Vertex AI client")
            return GenerativeModel(self.model_name, safety_settings=safety_settings)
        
        except Exception as e:
            logger.error(f"Error initializing model using `vertexai.init`: {e}")
            raise
    
    def _get_image_parts(self, image_files: Union[str, List[str]]) -> List[Part]:
        """
        Convert image files to a list of Part objects for multimodal content generation.
    
        Args:
            image_files (Union[str, List[str]]): Path to a single image file or a list 
                of paths to multiple image files. Supported formats include common image
                types like JPEG, PNG, GIF, etc.
        
        Returns:
            List[types.Part]: A list of Part objects containing the image data.
        """
        if not image_files:
            return []
        
        contents = []
        if isinstance(image_files, str):
            image_files = [image_files]
            
        for image_path in image_files:
            image_part = Part.from_image(Image.load_from_file(image_path))
            contents.append(image_part)
        return contents
    
    def _response_as_pydantic(
            self, 
            response:GenerationResponse, 
            response_format:Type[BaseModel]
            ):
        """
        Convert response from LLM client to the specified pydantic response format

        Args:
            response (GenerationResponse): The response generated by the LLm client
            response_format (Type[BaseModel]): Specified pydantic format
        """
        string_response = response.text.strip()
        if not string_response:
            logger.error("Empty or whitespace-only string")
            return None
        
        try:
            return response_format.model_validate_json(string_response)
        except Exception as e:
            logger.error(f"Failed to parse response from JSON string: {e}")
            return None
    
    def generate_response(
            self, 
            prompt: str, 
            response_format: Optional[Type[BaseModel]] = None, 
            images: Union[str, List[str]] = None
            ) -> Union[str,Type[BaseModel]]:
        """
        Generate a response using the native Vertex AI GenerativeModel.

        Args:
            prompt (str): The prompt to generate a response for.
            response_format (Optional[Type[BaseModel]]): Pydantic model for structured output.
            images (Optional[Union[str, List[str]]]): Image files to attach to prompt

        Returns:
            str: The generated response from the LLM.
        """
        contents = [prompt]
        if images:
            image_contents = self._get_image_parts(images)
            contents = contents + image_contents

        if response_format is None:
            # Regular text generation
            try:
                logger.info("Generating regular text response")
                response = self.llm.generate_content(contents)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error generating text response: {str(e)}")
                raise
        else:
            # Structured output generation
            try:
                # Convert Pydantic model to JSON schema
                json_schema = pydantic_to_vertex_schema(response_format)
                schema_str = json.dumps(json_schema, indent=2)
                    
                # Create generation config with response schema
                generation_config = GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=json_schema
                )

                logger.info("Generating structured response")
                response = self.llm.generate_content(
                    contents,
                    generation_config=generation_config
                )
                return self._response_as_pydantic(response, response_format)
                
            except Exception as e:
                logger.error(f"Error generating structured response: {str(e)}")
                # Fallback to regular generation with schema guidance
                logger.warning("Falling back to regular text generation with schema guidance")

                fallback_prompt = (
                    f"\n{prompt}\n\n"
                    "Please respond with a JSON object that matches this schema:"
                    f"\n\n{schema_str}\n\n"
                    "Ensure your response is valid JSON that conforms to the schema."
                )
                if images:
                    image_contents = self._get_image_parts(images)
                    fallback_contents = [fallback_prompt] + image_contents
                    response = self.llm.generate_content(
                        fallback_contents
                    )
                else:
                    response = self.llm.generate_content(
                        fallback_prompt
                    )
                return response.text.strip()


def test_handler_functionalities(handler: BaseLLMHandler, handler_name: str) -> None:
    """
    """
    # Define a structured schematic
    from pydantic import Field
    class Geography(BaseModel):      
        Location: str = Field(description="Name of the geographic location")
        Longitude: float = Field(description="Longitude")
        Latitude: float = Field(description="Latitude")

    # Test Handler for simple prompt
    logger.info(f"Testing {handler_name} Handler for simple prompt...")
    user_message = (
        "What is the capital of the United States of America? "
        "Provide only the geographic location name."
        "Besides the location name, please also return the longitude and latitude of the location"
    )
    try:
        response = handler.generate_response(prompt=user_message)
        logger.info(f"Query: {user_message}")
        logger.info(f"{handler_name} Response: {response}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

    # Test Handler for structured response on simple prompt
    logger.info(f"Testing {handler_name} Handler for structured response on simple prompt...")
    try:
        response = handler.generate_response(prompt=user_message, response_format=Geography)
        logger.info(f"Query: {user_message}")
        logger.info(f"{handler_name} Response: {response}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

    # Test Handler for prompt with images
    logger.info(f"Testing {handler_name} Handler for prompt with images...")
    user_message = (
        "Please analyze the image, including the background, displayed content, and any text, "
        "to determine which geographic area the map is depicting. Provide only the geographic location name."
        "Besides the area name, please also return the longitude and latitude of the image center"
    )
    try:
        response = handler.generate_response(prompt=user_message, images="input/images/figure-180-59.jpg")
        logger.info(f"Query: {user_message}")
        logger.info(f"{handler_name} Response: {response}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

    # Test Handler for structured response on images
    logger.info(f"Testing {handler_name} Handler for structured response on images...")
    try:
        response = handler.generate_response(prompt=user_message, images="input/images/figure-180-59.jpg", response_format=Geography)
        logger.info(f"Query: {user_message}")
        logger.info(f"{handler_name} Response: {response}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    return

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    
    # Test LlamaVertexHandler
    logger.info("Testing LlamaVertexHandler...")
    try:
        llama_client = LlamaVertexHandler(model_name="gemini-2.0-flash")
        test_handler_functionalities(handler=llama_client, handler_name="LlamaVertex")
    except Exception as e:
        logger.error(f"LlamaVertex handler test failed: {str(e)}")
    
    # Test VertexAIHandler
    logger.info("Testing VertexAIHandler...")
    try:
        vertex_client = VertexAIHandler(model_name="gemini-2.0-flash")
        test_handler_functionalities(handler=vertex_client, handler_name="VertexAI")
    except Exception as e:
        logger.error(f"VertexAI handler test failed: {str(e)}")