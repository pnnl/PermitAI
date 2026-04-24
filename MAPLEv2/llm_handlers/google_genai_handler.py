import os
import sys
from typing import Type, Optional, Dict, List, Union, Any
import mimetypes
import json
import time
from pydantic import BaseModel

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

try:
    from google import genai
    from google.genai import types
    from google.genai.types import CreateBatchJobConfig, GenerateContentConfig, GenerateContentResponse
    from google.cloud import storage
    from google.auth import default
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.genai is not available. GoogleGenAIHandler will not work.")

try:
    from llama_index.llms.google_genai import GoogleGenAI
except ImportError:
    print("Warning: Use `pip install llama-index-llms-google-genai` to install Google GenAI")

from utils.logging_utils import LogManager
from utils.schema_utils import pydantic_to_vertex_schema

if __name__ == "__main__":
    LogManager.initialize("logs/test_genai_handler.log")

logger = LogManager.get_logger("genai_handler")

class GoogleGenAIHandler:
    """
    Handler for Google Gemini using google.generativeai (GenAI) library with Client.
    
    This class provides an alternative implementation using the google.generativeai
    library with the Client interface.
    """
    def __init__(self, model_name: str, project_id: str = None, location: str = None, token_limit: int = 100000):
        """
        TODO: Needs review
        """
        self.model_name = model_name
        self.token_limit = token_limit

        # If VERTEXAI credentials are stored in a JSON file and specified as an environment variable, 
        # save it as an additional environment variable
        # this has been seen to work for Rounak
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.environ.get("VERTEXAI_CREDENTIALS_JSON_PATH"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("VERTEXAI_CREDENTIALS_JSON_PATH")
            logger.info("Using JSON credentials to load LLM handler")
        
        else:
            logger.info("JSON credentials file path is not present as an environment variable.")
            logger.info("If you have JSON credentials saved before, please save it as an environment variable")
            logger.info("Use: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/json/credentials/file'")
            logger.info("Attempting to load using Google Cloud Application Default Credentials")
            logger.info("Note that this will fail if you have not set Google ADC earlier")
            # Do nothing else
        
        # Let Google Auth library handle credential detection
        if not project_id:
            credentials, project_id = default()
        if not location:
            location = os.environ.get("VERTEXAI_LOCATION", "us-central1") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Initialize LLM client
        self.llm = self.get_llm(project_id=project_id, location=location)

        # Initialize RAG LLM client using llamaindex
        self.rag_llm = GoogleGenAI(
            model=model_name
            )
    
    def get_llm(self, project_id: str, location: str = "us-central1"):
        """
        Get the GenAI Client instance.

        Args:
            project_id (str): Name of the project ID for the service account
            location (str): Name of the service account location name. Default is 'us-central1'
        
        Returns:
            genai.Client: The GenAI Client instance.
        """
        try:
            # Initialize the GenAI Client
            client = genai.Client(
                vertexai=True, 
                project=project_id,
                location=location
                )
            logger.info(f"Google GenAI Client is successfully initialized.")
            return client
            
        except Exception as e:
            logger.error(f"Error initializing GenAI Client: {str(e)}")
            raise

    def _get_image_parts(self, image_files: Union[str, List[str]]) -> List[types.Part]:
        """
        Convert image files to a list of Part objects for multimodal content generation.
    
        Args:
            image_files (Union[str, List[str]]): Path to a single image file or a list 
                of paths to multiple image files. Supported formats include common image
                types like JPEG, PNG, GIF, etc.
        
        Returns:
            List[types.Part]: A list of Part objects containing the image data and 
                metadata. Each Part object includes the image bytes and its MIME type.
        """
        if not image_files:
            return []
        
        contents = []
        if isinstance(image_files, str):
            image_files = [image_files]
            
        for image_path in image_files:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type.startswith("image/"):
                logger.error(f"Determined MIME type '{mime_type}' for {image_path} does not seem to be an image.")
                raise ValueError(f"Invalid MIME type {mime_type} for image {image_path}")
            
            image_part = types.Part.from_bytes(
                data=image_bytes, mime_type=mime_type
            )
            contents.append(image_part)
        return contents
    
    def _response_as_pydantic(
            self, 
            response:GenerateContentResponse, 
            response_format:Type[BaseModel]
            ):
        """
        Convert response from LLM client to the specified pydantic response format

        Args:
            response (GenerateContentResponse): The response generated by the LLm client
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
        Generate a response using Google GenAI Client.
        
        Args:
            prompt (str): The prompt to generate a response for.
            response_format (Optional[Type[BaseModel]]): Pydantic model for structured output.
            images (Optional[Union[str, List[str]]]): Image files to attach to prompt
            
        Returns:
            str: The generated response from the model.
        """
        try:
            prompt = [prompt]
            if images:
                image_contents = self._get_image_parts(images)
                prompt = [prompt] + image_contents
            
            if response_format is None:
                # Regular text generation using Client
                response = self.llm.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # types.ThinkingConfig(include_thoughts=include_thoughts)
                        )
                )
                return response.candidates[0].content.parts[0].text.strip()
            else:
                # Try using generation config if supported
                try:
                    # Structured output generation
                    schema = pydantic_to_vertex_schema(response_format)
                    schema_str = json.dumps(schema, indent=2)
                    
                    generation_config = types.GenerateContentConfig(
                        response_mime_type = "application/json",
                        response_schema = schema
                    )
                    
                    response = self.llm.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=generation_config
                    )
                    return self._response_as_pydantic(response, response_format)
                    
                except Exception as config_error:
                    logger.warning(f"Generation config not supported, falling back to prompt-based schema: {config_error}")
                    
                    # Fallback to prompt-based structured generation
                    structured_prompt = (
                        f"{prompt}\n\n"
                        "Please respond with a JSON object that matches this schema:\n"
                        f"{schema_str}\n\n"
                        "Ensure your response is valid JSON that conforms to the schema."
                    )
                    
                    response = self.llm.models.generate_content(
                        model=self.model_name,
                        contents=structured_prompt
                    )
                    return response.text.strip()
                    
        except Exception as e:
            logger.error(f"Error generating response with GenAI Client: {str(e)}")
            raise

    def upload_image_to_gcs(self, image_path: str, bucket_name: str, gcs_image_directory: str, entry_id: str) -> str:
        """
        Upload image to GCS bucket and return the GCS URI.
        
        Args:
            image_path: Local path to the image
            bucket_name: GCS bucket name
            gcs_image_directory: GCS directory name for organizing uploads
            entry_id: unique id prefix for filename
            
        Returns:
            GCS URI for the uploaded image
        """
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create a unique blob name
            filename = os.path.basename(image_path)
            blob_name = f"{gcs_image_directory}/{entry_id}_{filename}"
            
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(image_path)
            
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            logger.info(f"Uploaded image {image_path} to {gcs_uri}")
            
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error uploading image {image_path} to GCS: {str(e)}")
            return None

    def batch_job(
            self, 
            ip_file_path: str,
            upload_file_path: str,
            op_file_path: str,
            bucket_name: str = "maple_evaluations",
            monitor_delay: int = 20
            ):
        """
        Args:
            ip_file_path (str): path for the batch inputs jsonl file
            upload_file_path (str): file path to save input jsonl file to gcp bucket
            op_file_path (str): gcp path for the batch outputs
            bucket_name (str): bucket name in GCP
        """
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            logger.info(f"Successfully created storage bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Error creating storage bucket: {e}")
            raise

        try:
            upload_blob = bucket.blob(upload_file_path)
            logger.info(f"Successfully created GCP blob: {upload_file_path}")

            upload_blob.upload_from_filename(ip_file_path)
            logger.info(f"Successfully uploaded prompts from {ip_file_path}")
        except Exception as e:
            logger.error(f"Error uploading prompts as JSONL file: {e}")
            raise


        # submit the job with the configurations
        logger.info(f"Source: gs://{bucket_name}/{upload_file_path}")
        logger.info(f"Destination: gs://{bucket_name}/{op_file_path}/")
        batch_job = self.llm.batches.create(
            model=self.model_name,
            src=f"gs://{bucket_name}/{upload_file_path}",
            config=CreateBatchJobConfig(
                dest=f"gs://{bucket_name}/{op_file_path}/",
                display_name="maple_evaluations"
                ),
        )

        job_name = batch_job.name
        logger.info(f"Created batch job: {job_name}")

        # Log progress of batch jobs. Check progress every 20 seconds
        while True:
            test_batch_job = self.llm.batches.get(name=job_name)
            logger.info(f"Batch Job Status: {test_batch_job.state}")
            if test_batch_job.state.name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
                logger.info(f"✅ Job completed with state: {test_batch_job.state.name}")
                break
            time.sleep(monitor_delay)

        # Read the predictions
        try:
            # Add delay to check for the predictions.jsonl
            time.sleep(monitor_delay)
            
            # Get the predictions
            predictions = self._load_latest_batch_predictions(
                bucket_name=bucket_name, 
                output_folder=op_file_path
            )
        
        except Exception as e:
            logger.error(f"Error extracting predictions: {e}")
            raise
        
        return predictions
    
    def _load_latest_batch_predictions(
            self, 
            bucket_name: str, 
            output_folder: str
            ) -> List[Dict[str, Any]]:
        """
        Load batch predictions from the latest timestamped prediction folder.
        
        Args:
            bucket_name (str): GCS bucket name 
            output_folder (str): Output folder name
            
        Returns:
            List of prediction results from predictions.jsonl in the latest folder
        """
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        
        # Build the base prefix
        prefix = f"{output_folder}/" if not output_folder.endswith('/') else output_folder
        
        logger.info(f"Searching for prediction folders in gs://{bucket_name}/{prefix}")
        
        # Get all blobs under the prefix and extract unique folder paths
        all_blobs = bucket.list_blobs(prefix=prefix)
        prediction_folders = set()
        
        for blob in all_blobs:
            # Extract the path relative to our prefix
            relative_path = blob.name[len(prefix):]
            
            # Check if this blob is in a prediction-* subfolder
            path_parts = relative_path.split('/')
            if len(path_parts) == 2 and path_parts[0].startswith('prediction-'):
                folder_path = prefix + path_parts[0] + '/'
                prediction_folders.add((folder_path, path_parts[0]))
        
        if not prediction_folders:
            logger.warning(f"No prediction folders found in gs://{bucket_name}/{prefix}")
            logger.warning("Available folders:")
            # Show what folders exist for debugging
            all_blobs = bucket.list_blobs(prefix=prefix)
            folders_seen = set()
            for blob in all_blobs:
                relative_path = blob.name[len(prefix):]
                if '/' in relative_path:
                    folder = relative_path.split('/')[0]
                    folders_seen.add(folder)
            
            for folder in sorted(folders_seen):
                logger.warning(f"  - {folder}")
            return []
        
        # Convert to list and sort by folder name to get the latest
        prediction_folder_list = [(path, name) for path, name in prediction_folders]
        latest_folder_path, latest_folder_name = sorted(prediction_folder_list, key=lambda x: x[1], reverse=True)[0]
        
        logger.info(f"Found {len(prediction_folders)} prediction folders")
        logger.info(f"Using latest: {latest_folder_name}")
        
        # Look for .jsonl files in the latest folder
        prediction_blobs = bucket.list_blobs(prefix=latest_folder_path)
        
        results = {}
        jsonl_files_found = 0
        
        for blob in prediction_blobs:
            if blob.name.endswith('.jsonl'):
                jsonl_files_found += 1
                logger.info(f"Loading predictions from: {blob.name}")
                
                try:
                    content = blob.download_as_string()
                    for line_num, line in enumerate(content.decode('utf-8').splitlines(), 1):
                        if line.strip():
                            try:
                                result = json.loads(line)
                                response_content = result["response"]["candidates"][0]["content"]["parts"]
                                text_content = '\n'.join([str(part["text"]) for part in response_content])
                                results[result["id"]] = text_content
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing line {line_num}: {e}")
                            except Exception as e:
                                logger.error(f"Error extracting response from {line_num}: {e}")
                                
                except Exception as e:
                    logger.error(f"Error loading {blob.name}: {e}")
        
        if jsonl_files_found == 0:
            logger.warning(f"No .jsonl files found in {latest_folder_path}")
            logger.warning("Files in latest folder:")
            for blob in bucket.list_blobs(prefix=latest_folder_path):
                logger.warning(f"  - {blob.name}")
        
        logger.info(f"Loaded {len(results)} predictions from {jsonl_files_found} files")
        return results
    
def test_handler_functionalities(handler: GoogleGenAIHandler, handler_name: str) -> None:
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

    # Test GoogleGenAIHandler
    logger.info(f"Testing GoogleGenAI Handler...")
    try:
        genai_client = GoogleGenAIHandler(model_name="gemini-2.0-flash")
        test_handler_functionalities(handler=genai_client, handler_name="GoogleGenAI")
    except Exception as e:
        logger.error(f"Initializing GoogleGenAI handler test failed: {str(e)}")