import os
import sys
from typing import Dict, Type, Optional, List, Union
import json
from pydantic import BaseModel
import mimetypes
import base64

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
from llm_handlers import BaseLLMHandler
from llama_index.llms import azure_openai
from openai import AzureOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from utils import LogManager
    
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
            "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-resource.openai.azure.com/"),
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY", None),
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            "engine": os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4-turbo-preview")
        }

    def get_llm(self)->AzureOpenAI:
        """
        Get an Azure OpenAI LLM client.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI LLM client.
        """
        llm = AzureOpenAI(
            azure_endpoint=self.client_kwargs["azure_endpoint"],
            api_key=self.client_kwargs["api_key"],
            api_version=self.client_kwargs["api_version"],
        )
        logger.info(f"Azure OpenAI (model name: {self.model_name}) is successfully connected.")
        return llm
    
    def get_rag_llm(self)->azure_openai.AzureOpenAI:
        """
        Get an Azure Llamaindex LLM RAG client.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI LLM client.
        """

        # Setup RAG LLM client for llama_index
        rag_llm = azure_openai.AzureOpenAI(
            azure_endpoint=self.client_kwargs["azure_endpoint"],
            api_key=self.client_kwargs["api_key"],
            api_version=self.client_kwargs["api_version"],
            azure_deployment=self.client_kwargs["engine"],
            model=self.model_name,
            max_tokens=self.token_limit
        )
        return rag_llm
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64 encoded image string.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the file is not a supported image format.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check if file is a supported image format
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            raise ValueError(f"File {image_path} is not a supported image format")
            
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"Successfully encoded image: {image_path}")
            return encoded_string
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
    
    def _get_image_mime_type(self, image_path: str) -> str:
        """
        Get the MIME type of an image file.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: MIME type of the image.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        return mime_type if mime_type else "image/jpeg"  # Default to JPEG if unknown
    
    def _create_image_content(self, image_path: str, detail: str = "high") -> Dict:
        """
        Create image content dictionary for the API.
        
        Args:
            image_path (str): Path to the image file.
            detail (str): Level of detail for image processing ("low", "high", "auto").
            
        Returns:
            Dict: Image content dictionary for the API.
        """
        encoded_image = self._encode_image(image_path)
        mime_type = self._get_image_mime_type(image_path)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded_image}",
                "detail": detail
            }
        }
    
    def _pydantic_to_json_schema(self, model_class: Type[BaseModel]) -> dict:
        """
        Convert Pydantic model to JSON schema format expected by Azure OpenAI.
        
        Args:
            model_class (Type[BaseModel]): Pydantic model class.
            
        Returns:
            dict: JSON schema for the model.
        """
        schema = model_class.model_json_schema()
        
        # Recursively ensure additionalProperties is false and required includes all properties
        def fix_schema_for_azure(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object" and "properties" in obj:
                    # Set additionalProperties to false
                    obj["additionalProperties"] = False
                    
                    # Ensure all properties are in the required array
                    all_properties = list(obj["properties"].keys())
                    obj["required"] = all_properties
                    
                # Recursively fix nested objects
                for key, value in obj.items():
                    if isinstance(value, dict):
                        fix_schema_for_azure(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                fix_schema_for_azure(item)
        
        # Apply the fix to the schema
        fix_schema_for_azure(schema)
        
        # Azure OpenAI expects a specific format for structured outputs
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", model_class.__name__).lower().replace(" ", "_"),
                "description": schema.get("description", f"Schema for {model_class.__name__}"),
                "schema": schema,
                "strict": True
            }
        }
    
    def _create_text_content(self, text: str) -> Dict:
        """
        Create text content dictionary for the API.
        
        Args:
            text (str): Text content.
            
        Returns:
            Dict: Text content dictionary for the API.
        """
        return {
            "type": "text",
            "text": text
        }
    
    def _prepare_messages(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        image_detail: str = "high",
        system_message: Optional[str] = None
    ) -> List[Dict]:
        """
        Prepare messages for the chat completion API.
        
        Args:
            prompt (str): Text prompt.
            images (Optional[List[str]]): List of image file paths.
            image_detail (str): Level of detail for image processing.
            system_message (Optional[str]): Optional system message.
            
        Returns:
            List[Dict]: Formatted messages for the API.
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Prepare user message content
        user_content = [self._create_text_content(prompt)]
        
        # Add images if provided
        if images:
            for image_path in images:
                try:
                    image_content = self._create_image_content(image_path, detail=image_detail)
                    user_content.append(image_content)
                    logger.info(f"Added image to prompt: {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to add image {image_path}: {str(e)}")
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def generate_response(
        self, 
        prompt: str, 
        images: Optional[Union[str, List[str]]] = None,
        image_detail: str = "high",
        system_message: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Generate a response using the Azure OpenAI LLM with optional image inputs and structured output.

        Args:
            prompt (str): The text prompt to generate a response for.
            images (Optional[Union[str, List[str]]]): Image file path(s) to include in the prompt.
            image_detail (str): Level of detail for image processing ("low", "high", "auto").
            system_message (Optional[str]): Optional system message to set context.
            response_format (Optional[Type[BaseModel]]): Pydantic model for structured output.

        Returns:
            Union[str, BaseModel]: The generated response from the LLM. 
                                 Returns a string if response_format is None, 
                                 otherwise returns an instance of the specified Pydantic model.
            
        Raises:
            Exception: If there's an error in generating the response.
            ValueError: If the structured response doesn't match the expected format.
        """
        try:
            # Handle single image input
            if isinstance(images, str):
                images = [images]
            
            # Prepare messages
            messages = self._prepare_messages(
                prompt=prompt,
                images=images,
                image_detail=image_detail,
                system_message=system_message
            )
            
            # Prepare API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages
            }
            
            # Add structured output format if specified
            if response_format is not None:
                json_schema = self._pydantic_to_json_schema(response_format)
                api_params["response_format"] = json_schema
                logger.info(f"Using structured output with schema: {response_format.__name__}")
            
            # Make the API call
            response = self.llm.chat.completions.create(**api_params)
            
            # Extract the response text
            response_text = response.choices[0].message.content
            logger.info("Successfully generated response")
            
            if response_text is None:
                response_text = ""
            
            # Handle structured output
            if response_format is not None:
                try:
                    # Parse the JSON response
                    response_data = json.loads(response_text)
                    
                    # Create and validate the Pydantic model instance
                    structured_response = response_format(**response_data)
                    logger.info(f"Successfully created structured response of type {response_format.__name__}")
                    return structured_response
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Response text: {response_text}")
                    raise ValueError(f"Invalid JSON in structured response: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"Failed to create {response_format.__name__} instance: {str(e)}")
                    logger.error(f"Response data: {response_text}")
                    raise ValueError(f"Response doesn't match expected schema: {str(e)}")
            
            # Return plain text response
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise


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
    
    def get_rag_llm(self):
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    logger.info("Loading AZURE credentials as environment variables.")
    llm_client = AzureOpenAIHandler(model_name="gpt-4-turbo-preview")

    class CapitalInfo(BaseModel):
        country: str
        capital: str
        population: Optional[int] = None
        interesting_fact: str

    class ImageAnalysis(BaseModel):
        """Schema for image analysis response."""
        objects_detected: List[str]
        scene_description: str
        colors_identified: List[str]
        estimated_confidence: float
    
    structured_response = llm_client.generate_response(
        prompt="Tell me about the capital of France",
        response_format=CapitalInfo
    )
    print(f"Structured response: {structured_response}")
    print(f"Capital: {structured_response.capital}")
    print(f"Country: {structured_response.country}")

    # Example with image and structured output
    image_analysis = llm_client.generate_response(
        prompt="Analyze this image and provide structured information about what you see.",
        images=["input/images/figure-180-59.jpg"],
        response_format=ImageAnalysis
    )
    print(f"Image analysis: {image_analysis}")
    print(f"Objects detected: {image_analysis.objects_detected}")
    print(f"Scene: {image_analysis.scene_description}")
