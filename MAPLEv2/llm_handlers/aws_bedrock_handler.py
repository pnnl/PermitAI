import os
import sys
import boto3
import base64
import json
from typing import Dict, List, Union, Optional, Type, Any
from PIL import Image

import mimetypes
from pydantic import BaseModel
from instructor import from_bedrock
import instructor

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers import BaseLLMHandler
from llama_index.llms.bedrock import Bedrock
from utils import LogManager
from utils.image_utils import smart_image_optimization

if __name__ == "__main__":
    LogManager.initialize("logs/test_aws_handler.log")
    
logger = LogManager.get_logger("aws_handler")

class AWSBedrockHandler(BaseLLMHandler):
    """
    Enhanced handler for AWS Bedrock LLM service with vision and structured output capabilities.

    This class extends BaseLLMHandler to provide specific implementation
    for AWS Bedrock LLM service with support for images and structured JSON output.
    """ 
        
    def get_external_credentials(self):
        """
        Retrieve AWS credentials for authentication.

        Returns:
            dict: A dictionary containing AWS credentials.
        """
        session_kwargs = {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            "profile_name": os.environ.get("AWS_PROFILE", "PowerUserAccess-776026895843")
            }
        
        session_kwargs['aws_access_key_id'] = os.environ.get("AWS_ACCESS_KEY_ID", None)
        session_kwargs['aws_secret_access_key'] = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        session_kwargs['aws_session_token'] = os.environ.get("AWS_SESSION_TOKEN", None)
        logger.info(f"Using profile: {session_kwargs['profile_name']}")
        
        session = boto3.Session(**session_kwargs)
        logger.info(f"Create new client using region: {session_kwargs['region_name']}")
        try:
            credentials = session.get_credentials().get_frozen_credentials()
            logger.info(f"Frozen credentials obtained for profile: {session_kwargs['profile_name']}")
        except Exception as e:
            logger.error(e)
            logger.error("Configure sso using `aws configure sso`")
    
        return {
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
            "aws_session_token": credentials.token,
            "aws_session_expiration": None,
            "aws_region_name": session_kwargs["region_name"],
        }

    def get_llm(self)->Bedrock:
        """
        Get an AWS Bedrock LLM client.

        Returns:
            Bedrock: An instance of the Bedrock LLM client.
        """
        llm = Bedrock(
            model=self.model_name,
            aws_access_key_id=self.client_kwargs["aws_access_key_id"],
            aws_secret_access_key=self.client_kwargs["aws_secret_access_key"],
            aws_session_token=self.client_kwargs["aws_session_token"],
            region_name=self.client_kwargs["aws_region_name"],
            context_size=self.token_limit
        )
        logger.info(f"AWS cloud (model name: {self.model_name}) is successfully connected.")

        # Also create a direct Bedrock runtime client for advanced features
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=self.client_kwargs["aws_access_key_id"],
            aws_secret_access_key=self.client_kwargs["aws_secret_access_key"],
            aws_session_token=self.client_kwargs["aws_session_token"],
            region_name=self.client_kwargs["aws_region_name"]
        )
        return llm
    
    def get_rag_llm(self)->Bedrock:
        """
        Get an AWS Bedrock LLM client.

        Returns:
            Bedrock: An instance of the Bedrock LLM client.
        """
        return Bedrock(
            model=self.model_name,
            aws_access_key_id=self.client_kwargs["aws_access_key_id"],
            aws_secret_access_key=self.client_kwargs["aws_secret_access_key"],
            aws_session_token=self.client_kwargs["aws_session_token"],
            region_name=self.client_kwargs["aws_region_name"],
            context_size=self.token_limit
        )
    
    def _validate_image_for_claude(self, image_path: str) -> tuple:
        """
        Enhanced validation for image dimensions and file size with detailed analysis.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            tuple: (is_valid, error_message, optimization_needed, optimization_type)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)
                file_size_mb = file_size / (1024 * 1024)
                
                # Claude's constraints (conservative limits)
                MAX_DIMENSION = 7500  # pixels
                MAX_FILE_SIZE_MB = 4.5  # MB (conservative, actual limit is 5MB)
                
                issues = []
                optimization_needed = False
                optimization_type = []
                
                if width > MAX_DIMENSION or height > MAX_DIMENSION:
                    issues.append(f"Dimensions ({width}x{height}) exceed limit ({MAX_DIMENSION}px)")
                    optimization_needed = True
                    optimization_type.append("resize")
                
                if file_size_mb > MAX_FILE_SIZE_MB:
                    issues.append(f"File size ({file_size_mb:.2f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)")
                    optimization_needed = True
                    optimization_type.append("compress")
                
                # Additional checks for image characteristics
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    issues.append(f"Unsupported color mode: {img.mode}")
                    optimization_needed = True
                    optimization_type.append("convert")
                
                if issues:
                    return False, "; ".join(issues), optimization_needed, optimization_type
                    
                return True, "Image meets all requirements", False, []
                
        except Exception as e:
            return False, f"Could not validate image: {str(e)}", False, []
    
    def _encode_image_for_claude(self, image_path: str) -> str:
        """
        Encode an image file to base64 string with comprehensive optimization for Claude compatibility.
        Uses temporary files and cleans them up automatically.
        
        Args:
            image_path (str): Path to the original image file.
            
        Returns:
            str: Base64 encoded image string.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the file is not a supported image format.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check if file is a supported image format
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            logger.error(f"File {image_path} is not a supported image format")
            raise ValueError(f"File {image_path} is not a supported image format")
        
        temp_file_path = None
        try:
            # Enhanced validation
            is_valid, message, needs_optimization, optimization_types = self._validate_image_for_claude(image_path)
            
            if not is_valid:
                if needs_optimization:
                    logger.warning(f"Image needs optimization: {message}")
                    logger.info(f"Applying optimizations: {', '.join(optimization_types)}")
                    
                    # Use smart optimization that handles multiple issues
                    encoded_string, temp_file_path = smart_image_optimization(image_path)
                    logger.info(f"Successfully optimized and encoded image: {image_path}")
                    return encoded_string
                else:
                    raise ValueError(f"Image cannot be processed: {message}")
            
            # Image is valid, proceed with normal encoding
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"Successfully encoded image without optimization: {image_path}")
            return encoded_string
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
        finally:
            # Clean up temporary file if it was created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")

    def _encode_image_for_pixtral(self, image_path: str) -> str:
        """
        Encode an image file to base64 string with comprehensive optimization for Claude compatibility.
        Uses temporary files and cleans them up automatically.
        
        Args:
            image_path (str): Path to the original image file.
            
        Returns:
            str: Base64 encoded image string.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the file is not a supported image format.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check if file is a supported image format
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            logger.error(f"File {image_path} is not a supported image format")
            raise ValueError(f"File {image_path} is not a supported image format")
        
        try:
            # Image Encoding
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"Successfully encoded image without optimization: {image_path}")
            return encoded_string
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
        
    
    def _get_image_media_type(self, image_path: str) -> str:
        """
        Get the media type of an image file for Bedrock.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Media type of the image.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        # Bedrock supports specific media types
        if mime_type in ['image/jpeg', 'image/jpg']:
            return 'image/jpeg'
        elif mime_type == 'image/png':
            return 'image/png'
        elif mime_type == 'image/gif':
            return 'image/gif'
        elif mime_type == 'image/webp':
            return 'image/webp'
        else:
            return 'image/jpeg'  # Default fallback
    
    def _create_anthropic_messages(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create message format for Anthropic Claude models on Bedrock.
        
        Args:
            prompt (str): Text prompt.
            images (Optional[List[str]]): List of image file paths.
            system_message (Optional[str]): Optional system message.
            
        Returns:
            Dict[str, Any]: Formatted message for Anthropic models.
        """
        # Anthropic format for Bedrock
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": []
        }
        
        # Add system message if provided
        if system_message:
            body["system"] = system_message
        
        # Create user message content
        user_content = [{"type": "text", "text": prompt}]
        
        # Add images if provided
        if images:
            successful_images = 0
            for image_path in images:
                try:
                    # This will automatically resize if needed
                    encoded_image = self._encode_image_for_claude(image_path)
                    media_type = self._get_image_media_type(image_path)
                    
                    image_content = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image
                        }
                    }
                    user_content.append(image_content)
                    successful_images += 1
                    logger.info(f"Added image to Anthropic message: {image_path}")
                
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {str(e)}")
                    # Continue with other images rather than failing completely
                    continue

            if successful_images == 0 and images:
                logger.warning("No images could be processed successfully")
            else:
                logger.info(f"Successfully processed {successful_images} out of {len(images)} images")
        
        body["messages"] = [{"role": "user", "content": user_content}]
        return body
    
    def _create_pixtral_messages(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create message format for Mistral Pixtral models on Bedrock.
        
        Args:
            prompt (str): Text prompt.
            images (Optional[List[str]]): List of image file paths.
            system_message (Optional[str]): Optional system message.
            
        Returns:
            Dict[str, Any]: Formatted message for Mistral models.
        """
        # Mistral format for Bedrock
        body = {
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": []
        }
        
        # Add system message if provided
        if system_message:
            body["system"] = system_message
        
        # Create user message content
        user_content = [{"type": "text", "text": prompt}]
        
        # Add images if provided
        if images:
            successful_images = 0
            for image_path in images:
                try:
                    # Encode image for pixtral message
                    encoded_image = self._encode_image_for_pixtral(image_path)
                    
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                    user_content.append(image_content)
                    successful_images += 1
                    logger.info(f"Added image to Mistral message: {image_path}")
                
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {str(e)}")
                    # Continue with other images rather than failing completely
                    continue

            if successful_images == 0 and images:
                logger.warning("No images could be processed successfully")
            else:
                logger.info(f"Successfully processed {successful_images} out of {len(images)} images")
        
        body["messages"] = [{"role": "user", "content": user_content}]
        return body
    
    def _read_image_file(self, file_path: str) -> bytes:
        """Read a file in binary mode."""
        try:
            with open(file_path, 'rb') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def _create_llama4_messages(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create message format for Meta Llama models on Bedrock.
        
        Args:
            prompt (str): Text prompt.
            images (Optional[List[str]]): List of image file paths.
            system_message (Optional[str]): Optional system message.
            
        Returns:
            Dict[str, Any]: Formatted message for Meta models.
        """
        # Llama format for Bedrock
        body = {
            "max_tokens": 4096,
            "temperature": 0.0,
            "messages": []
        }
        
        # Initialize content list
        user_content = []

        # For Meta models, we'll primarily use text with image descriptions
        full_prompt = prompt
        
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {full_prompt}"

        user_content.append({"text": full_prompt})
        
        # Add images if provided
        if images:
            successful_images = 0
            for image_path in images:
                try:
                    image_content = {
                        "image": {
                            "format": self._get_image_media_type(image_path).split('/')[-1],
                            "source": {"bytes": self._read_image_file(image_path)}
                        }
                    }
                    user_content.append(image_content)
                    successful_images += 1
                    logger.info(f"Added image to Llama message: {image_path}")
                
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {str(e)}")
                    # Continue with other images rather than failing completely
                    continue

            if successful_images == 0 and images:
                logger.warning("No images could be processed successfully")
            else:
                logger.info(f"Successfully processed {successful_images} out of {len(images)} images")
        
        body["messages"] = [{"role": "user", "content": user_content}]
        return body
    
    def _create_llama3_messages(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create message format for Meta Llama models on Bedrock.
        
        Args:
            prompt (str): Text prompt.
            images (Optional[List[str]]): List of image file paths.
            system_message (Optional[str]): Optional system message.
            
        Returns:
            Dict[str, Any]: Formatted message for Meta models.
        """
        # Ignore images in the prompt
        if images:
            logger.warning(f"Model {self.model_name} does not support vision. Images will be ignored.")
        
        # Initialize content list
        user_content = []

        # For Meta models, we'll primarily use text with image descriptions
        full_prompt = prompt
        
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {full_prompt}"

        user_content.append({"text": full_prompt})

        # Llama3 format for Bedrock
        body = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
        return body
    
    def _pydantic_to_json_schema(self, model_class: Type[BaseModel]) -> Dict:
        """
        Convert Pydantic model to JSON schema for structured output.
        
        Args:
            model_class (Type[BaseModel]): Pydantic model class.
            
        Returns:
            Dict: JSON schema for the model.
        """
        return model_class.model_json_schema()
    
    def _add_structured_output_instructions(self, prompt: str, response_format: Type[BaseModel]) -> str:
        """
        Add structured output instructions to the prompt.
        
        Args:
            prompt (str): Original prompt.
            response_format (Type[BaseModel]): Pydantic model for structured output.
            
        Returns:
            str: Enhanced prompt with structured output instructions.
        """
        schema = self._pydantic_to_json_schema(response_format)
        schema_str = json.dumps(schema, indent=2)
        
        enhanced_prompt = f"""{prompt}

Please respond with a valid JSON object that matches this exact schema:

{schema_str}

Ensure your response is valid JSON that conforms to the schema. Do not include any text outside the JSON object."""
        
        return enhanced_prompt
    
    def generate_response(
        self, 
        prompt: str, 
        images: Optional[Union[str, List[str]]] = None,
        system_message: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Generate a response using AWS Bedrock with optional image inputs and structured output.

        Args:
            prompt (str): The text prompt to generate a response for.
            images (Optional[Union[str, List[str]]]): Image file path(s) to include in the prompt.
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
            
            # Modify prompt for structured output
            final_prompt = prompt
            if response_format is not None:
                final_prompt = self._add_structured_output_instructions(prompt, response_format)
            
            # Log the request details
            image_count = len(images) if images else 0
            structured_output = response_format is not None
            logger.info(f"Generating response with {image_count} images, "
                       f"structured_output={structured_output}")
            
            # Check if model supports vision (mainly Claude models, Pixtral models, Llama models)
            
            # Claude models
            if any(model in self.model_name.lower() for model in ['claude', 'anthropic']):
                try:
                    body = self._create_anthropic_messages(final_prompt, images, system_message)
                    
                    response = self.bedrock_runtime.invoke_model(
                        modelId=self.model_name,
                        body=json.dumps(body),
                        contentType='application/json'
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    response_text = response_body.get('content', [{}])[0].get('text', '')
                    
                except Exception as e:
                    logger.error(f"Error generating response from anthropic model: {e}")
                    raise
            
            # Pixtral models
            elif any(model in self.model_name.lower() for model in ['pixtral']):
                try:
                    body = self._create_pixtral_messages(final_prompt, images, system_message)
                    
                    response = self.bedrock_runtime.invoke_model(
                        modelId=self.model_name,
                        body=json.dumps(body),
                        contentType='application/json'
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    response_text = response_body.get('choices', [{}])[0].get('message', '').get('content','```json\n```')
                    
                except Exception as e:
                    logger.error(f"Error generating response from pixtral model: {e}")
                    raise

            # Llama 4 model
            elif any(model in self.model_name.lower() for model in ['llama4']):
                try:
                    body = self._create_llama4_messages(final_prompt, images, system_message)
                    
                    response = self.bedrock_runtime.converse(
                        modelId=self.model_name,
                        messages=body["messages"]
                    )
                    
                    response_text = response["output"]["message"]["content"][-1]["text"]
                    
                except Exception as e:
                    logger.error(f"Error generating response from llama4 model: {e}")
                    raise
            
            # Llama 3 model
            elif any(model in self.model_name.lower() for model in ['llama3']):
                try:
                    body = self._create_llama3_messages(final_prompt, images, system_message)
                    
                    if response_format is not None:
                        instructor_client = from_bedrock(client=self.bedrock_runtime)
                        logger.info("Created instructor client for structured response")

                        try:
                            structured_response = instructor_client.chat.completions.create(
                                model=self.model_name,
                                response_model=response_format,
                                messages=body["messages"]
                            )
                            
                            # Return the structured response
                            return structured_response
                        
                        except Exception as e:
                            logger.error(f"Error generating response from instructor module: {e}")
                            raise
                    
                    else:
                        response = self.bedrock_runtime.converse(
                            modelId=self.model_name,
                            messages=body["messages"],
                            inferenceConfig={
                                "maxTokens":body['max_tokens'], 
                                "temperature": body['temperature']
                                }
                        )
                    
                        response_text = response["output"]["message"]["content"][-1]["text"]
                        
                        # Return plain text response
                        return response_text
                    
                except Exception as e:
                    logger.error(f"Error generating response from llama3 model: {e}")
                    raise

            else:
                if images:
                    logger.warning(f"Model {self.model_name} may not support vision. Images will be ignored.")

                # For text-only, we can use the standard LlamaIndex interface
                if system_message:
                    full_prompt = f"System: {system_message}\n\nUser: {final_prompt}"
                else:
                    full_prompt = final_prompt
                
                response = self.llm.complete(full_prompt)
                response_text = response.text.strip()
            
            logger.info("Successfully generated response")
            
            # Handle structured output
            if response_format is not None:
                try:
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        response_data = json.loads(json_str)
                    else:
                        # Fallback: try to parse the entire response as JSON
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
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(".env")
    
    handler = AWSBedrockHandler(model_name="anthropic.claude-3-5-sonnet-20241022-v2:0")
    
    # Text-only example
    text_response = handler.generate_response("What is the capital of France?")
    print(f"Text response: {text_response}")
    
    # Example with structured output (no images)
    class CapitalInfo(BaseModel):
        country: str
        capital: str
        population: Optional[int] = None
        interesting_fact: str
    
    structured_response = handler.generate_response(
        prompt="Tell me about the capital of France",
        response_format=CapitalInfo
    )
    print(f"Structured response: {structured_response}")
    print(f"Capital: {structured_response.capital}")
    print(f"Country: {structured_response.country}")

    # Example with image and structured output (uncomment and provide actual image path)
    class ImageAnalysis(BaseModel):
        """Schema for image analysis response."""
        objects_detected: List[str]
        scene_description: str
        text_content: Optional[str] = None
        estimated_confidence: float

    image_analysis = handler.generate_response(
        prompt="Analyze this image and provide structured information about what you see.",
        images=["input/images/figure-180-59.jpg"],
        response_format=ImageAnalysis
    )
    print(f"Image analysis: {image_analysis}")
    print(f"Objects detected: {image_analysis.objects_detected}")
    print(f"Scene: {image_analysis.scene_description}")
    