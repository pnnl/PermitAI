# LLM Handlers

The `llm_handlers` directory contains implementations for various Language Model providers. Each handler follows a common interface defined in `base_handler.py` while providing provider-specific functionality.

## Contents

1. [Base Handler](#base-handler)
2. [Provider Specific Handlers](#provider-specific-handlers)
    - [Azure OpenAI Handler](#azure_openai_handlerpy)
    - [AWS Bedrock Handler](#aws_bedrock_handlerpy)
    - [Vertex Handler](#vertex_gemini_handlerpy)
    - [HuggingFace Handler](#huggingface_handlerpy)
3. [Common Usage Patterns](#common-usage-patterns)
4. [Configuration Best Practices](#configuration-best-practices)
5. [Contributing](#contributing)

## Base Handler

### `base_handler.py`
The base abstract class that defines the interface for all LLM handlers.

```python
from llm_handlers.base_handler import BaseLLMHandler

class CustomHandler(BaseLLMHandler):
    def get_external_credentials(self):
        # Implementation
        pass

    def get_llm(self):
        # Implementation
        pass

    def generate_response(self, prompt):
        # Implementation
        pass
```

Required methods:
- `get_external_credentials()`: Retrieves API credentials
- `get_llm()`: Initializes and returns an LLM client
- `generate_response(prompt)`: Generates responses to prompts

## Provider-Specific Handlers

### `azure_openai_handler.py`
Handler for Azure OpenAI services, supporting GPT models.

```python
from llm_handlers.azure_openai_handler import AzureOpenAIHandler

# Regular handler for response generation
handler = AzureOpenAIHandler("gpt-4-turbo-preview")

# Chat-optimized handler for RAGAS evaluation
chat_handler = AzureChatOpenAIHandler("gpt-4-turbo-preview")
```

Required Environment Variables:
```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2023-12-01-preview"
AZURE_DEPLOYMENT_NAME="gpt-4-turbo-preview"
```

### `aws_bedrock_handler.py`
Handler for AWS Bedrock services, supporting models like Anthropic Claude and Mistral.

```python
from llm_handlers.aws_bedrock_handler import AWSBedrockHandler

# Using AWS role
handler = AWSBedrockHandler(
    model_name="mistral.mistral-7b-instruct-v0:2",
    aws_role_use=True
)

# Using AWS profile
handler = AWSBedrockHandler(
    model_name="anthropic.claude-v2",
    aws_role_use=False
)
```

Required Environment Variables:
```bash
AWS_DEFAULT_REGION="us-west-2"
AWS_PROFILE="your-profile"  # If not using role
AWS_BEDROCK_ROLE="arn:aws:iam::your-role"  # If using role
```

### `vertex_gemini_handler.py`
Handler for Google Vertex AI's Gemini models.

```python
from llm_handlers.vertex_gemini_handler import VertexGeminiHandler

handler = VertexGeminiHandler("gemini-1.5-pro-preview-0409")
```

Required Environment Variables:
```bash
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/credentials.json"
```

Safety Settings:
```python
# Default safety settings (disabled for evaluation)
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE
}
```

### `huggingface_handler.py`
Handler for local Hugging Face models and pipeline setup.

```python
from llm_handlers.huggingface_handler import HuggingFaceHandler

handler = HuggingFaceHandler(
    model_path="/path/to/local/model"
)
```

Features:
- Automatic GPU detection and utilization
- Pipeline configuration for text generation
- Customizable generation parameters

Generation Parameters:
```python
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
```

## Common Usage Patterns

### Basic Response Generation
```python
# Initialize handler
handler = AzureOpenAIHandler("gpt-4-turbo-preview")

# Generate response
response = handler.generate_response("What is the capital of France?")
```

### Evaluation Setup
```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator(
    llm_handler=handler,
    docID_csvpath="metadata.csv",
    embed_model="BAAI/bge-small-en-v1.5",
    prompt_file="prompts/template.txt"
)
```

### Error Handling
All handlers implement robust error handling:
```python
try:
    response = handler.generate_response(prompt)
except Exception as e:
    logger.error(f"Error generating response: {str(e)}")
    raise
```

## Configuration Best Practices

1. Use environment variables for credentials:
```bash
# .env file
AZURE_OPENAI_API_KEY="your-key"
AWS_PROFILE="your-profile"
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/creds.json"
```

2. Load environment variables:
```python
from dotenv import load_dotenv
load_dotenv()
```

3. Configure logging:
```python
from utils.logging_utils import LogManager
logger = LogManager.get_logger(__name__)
```

## Handler Selection Guide

Choose the appropriate handler based on your needs:

1. **Azure OpenAI Handler**
   - Best for: Production deployments requiring high reliability
   - Models: GPT-4
   - Features: Rate limiting, automatic retries

2. **AWS Bedrock Handler**
   - Best for: AWS infrastructure integration
   - Models: Claude, Mistral, Llama3
   - Features: IAM role support, region configuration

3. **Vertex Gemini Handler**
   - Best for: Google Cloud integration
   - Models: Gemini Pro
   - Features: Safety settings, context management

4. **Hugging Face Handler**
   - Best for: Local deployment, custom models
   - Models: Any HuggingFace model
   - Features: GPU utilization, pipeline customization

## Contributing

To add a new handler:

1. Create a new file `your_handler.py`
2. Inherit from `BaseLLMHandler`
3. Implement required methods
4. Add appropriate error handling and logging
5. Include documentation and usage examples