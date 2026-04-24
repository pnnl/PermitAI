# LLM Handlers

The `llm_handlers` directory provides a unified interface for multiple LLM providers. All handlers implement the abstract `BaseLLMHandler` interface and expose a consistent `generate_response` API across text generation, structured output, and multimodal (image) inputs.

## Contents

1. [Base Handler](#base-handler)
2. [Provider Handlers](#provider-handlers)
   - [Azure OpenAI](#azure_openai_handlerpy)
   - [AWS Bedrock](#aws_bedrock_handlerpy)
   - [Vertex AI](#vertex_gemini_handlerpy)
   - [Google GenAI](#google_genai_handlerpy)
   - [HuggingFace](#huggingface_handlerpy)
3. [Capability Matrix](#capability-matrix)
4. [Usage](#usage)
5. [Adding a New Handler](#adding-a-new-handler)

---

## Base Handler

### `base_handler.py`

Abstract base class that all handlers must implement.

```python
from llm_handlers.base_handler import BaseLLMHandler

class MyHandler(BaseLLMHandler):
    def get_external_credentials(self) -> dict: ...
    def get_llm(self): ...
    def get_rag_llm(self): ...
    def generate_response(self, prompt: str, response_format=None) -> str: ...
```

**Constructor**: `BaseLLMHandler(model_name, token_limit=100000)`  
On init, automatically calls `get_external_credentials()`, `get_llm()`, and `get_rag_llm()`.

**Attributes**:

| Attribute | Type | Description |
|---|---|---|
| `model_name` | `str` | Model identifier |
| `token_limit` | `int` | Maximum prompt token length |
| `client_kwargs` | `dict` | Credentials returned by `get_external_credentials()` |
| `llm` | object | Primary LLM client |
| `rag_llm` | object | LlamaIndex-compatible client for RAG pipelines |

---

## Provider Handlers

### `azure_openai_handler.py`

Two classes for Azure OpenAI services.

**`AzureOpenAIHandler`** — standard handler using the OpenAI SDK.

```python
from llm_handlers.azure_openai_handler import AzureOpenAIHandler

handler = AzureOpenAIHandler("gpt-4-turbo-preview")
response = handler.generate_response("Summarize this document.", images=["figure.jpg"])
```

**`AzureChatOpenAIHandler`** — wraps a LangChain `AzureChatOpenAI` client; intended for RAGAS scoring pipelines. `get_rag_llm()` returns `None`.

Required environment variables:
```bash
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_API_VERSION="2023-12-01-preview"
AZURE_DEPLOYMENT_NAME="your-deployment-name"
```

---

### `aws_bedrock_handler.py`

**`AWSBedrockHandler`** — handler for AWS Bedrock with model-specific message formatting.

```python
from llm_handlers.aws_bedrock_handler import AWSBedrockHandler

# Using IAM role
handler = AWSBedrockHandler("anthropic.claude-3-5-sonnet-20241022-v2:0", aws_role_use=True)

# Using AWS profile
handler = AWSBedrockHandler("mistral.mistral-7b-instruct-v0:2", aws_role_use=False)
```

Supported model families and their message formats:

| Model family | Format method |
|---|---|
| Anthropic Claude | `_create_anthropic_messages` |
| Mistral Pixtral | `_create_pixtral_messages` |
| Meta Llama 4 | `_create_llama4_messages` |
| Meta Llama 3 | `_create_llama3_messages` (text-only) |

Required environment variables:
```bash
AWS_DEFAULT_REGION="us-west-2"
AWS_PROFILE="your-profile"        # if not using role
AWS_BEDROCK_ROLE="arn:aws:iam::your-role"  # if using role
```

---

### `vertex_gemini_handler.py`

Two classes for Google Vertex AI.

**`LlamaVertexHandler`** — text-only handler using LlamaIndex's Vertex client. Raises `ValueError` if `response_format` or `images` are passed.

**`VertexAIHandler`** (extends `LlamaVertexHandler`) — full-featured handler using the native `vertexai` SDK. Supports structured output and image inputs. Falls back to prompt-based schema guidance if `GenerationConfig` is unsupported.

```python
from llm_handlers.vertex_gemini_handler import VertexAIHandler, LlamaVertexHandler

# Text only
llama_handler = LlamaVertexHandler("gemini-2.5-pro")

# Text + images + structured output
vertex_handler = VertexAIHandler("gemini-2.5-pro")
```

Safety settings are set to `BLOCK_NONE` for all harm categories by default (required for evaluation use cases).

Required environment variables:
```bash
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/service-account.json"
VERTEXAI_LOCATION="us-central1"   # optional, defaults to us-central1
```

---

### `google_genai_handler.py`

**`GoogleGenAIHandler`** — handler using the `google.genai` Client API (Vertex AI backend). Does **not** inherit from `BaseLLMHandler`; initializes `llm` (genai.Client) and `rag_llm` (LlamaIndex GoogleGenAI) directly in `__init__`.

```python
from llm_handlers.google_genai_handler import GoogleGenAIHandler

handler = GoogleGenAIHandler("gemini-2.5-pro")
response = handler.generate_response(prompt, response_format=MyPydanticModel)
```

Also supports GCS-based batch inference:

```python
predictions = handler.batch_job(
    ip_file_path="local/inputs.jsonl",
    upload_file_path="gcs/path/inputs.jsonl",
    op_file_path="gcs/path/outputs/",
    bucket_name="my-bucket"
)
```

Credential resolution order:
1. `VERTEXAI_CREDENTIALS_JSON_PATH` → sets `GOOGLE_APPLICATION_CREDENTIALS`
2. Google Application Default Credentials (ADC)

Required environment variables:
```bash
VERTEXAI_CREDENTIALS_JSON_PATH="/path/to/service-account.json"
# or
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

VERTEXAI_LOCATION="us-central1"   # optional
```

---

### `huggingface_handler.py`

**`HuggingFaceHandler`** — loads a local HuggingFace model and runs inference via a `transformers` text-generation pipeline. Automatically uses GPU if available.

```python
from llm_handlers.huggingface_handler import HuggingFaceHandler

handler = HuggingFaceHandler(
    model_path="/path/to/local/model",
    adaptor_path="/path/to/lora/adaptor"  # optional
)
response = handler.generate_response("What is NEPA?")
```

Generation parameters:
```python
max_new_tokens = 2000
return_full_text = False
temperature = 0.001
```

**`PipelineLLM`** — thin LlamaIndex-compatible wrapper around the pipeline, used internally as `rag_llm`.

No environment variables required. Model path is passed directly to the constructor.

---

## Capability Matrix

| Handler | Text | Structured output | Images | RAG (`rag_llm`) | Batch |
|---|:---:|:---:|:---:|:---:|:---:|
| `AzureOpenAIHandler` | ✓ | ✓ | ✓ | ✓ | |
| `AzureChatOpenAIHandler` | ✓ | | | | |
| `AWSBedrockHandler` | ✓ | ✓ | ✓ (Claude, Pixtral, Llama4) | ✓ | |
| `LlamaVertexHandler` | ✓ | | | ✓ | |
| `VertexAIHandler` | ✓ | ✓ | ✓ | ✓ | |
| `GoogleGenAIHandler` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HuggingFaceHandler` | ✓ | | | ✓ | |

---

## Usage

### Basic text generation

```python
from llm_handlers.vertex_gemini_handler import VertexAIHandler

handler = VertexAIHandler("gemini-2.5-pro")
response = handler.generate_response("Describe the proposed action in this document.")
```

### Structured output

```python
from pydantic import BaseModel, Field

class ProjectInfo(BaseModel):
    title: str = Field(description="Project title")
    location: str = Field(description="Project location")

response = handler.generate_response(prompt, response_format=ProjectInfo)
print(response.title, response.location)
```

### Multimodal (images)

```python
response = handler.generate_response(
    prompt="What geographic area does this map show?",
    images=["figure-180-59.jpg", "figure-180-60.jpg"]
)
```

### Loading credentials via `.env`

```python
from dotenv import load_dotenv
load_dotenv(".env")
```

### Logging

```python
from utils.logging_utils import LogManager
logger = LogManager.get_logger(__name__)
```

---

## Adding a New Handler

1. Create `your_handler.py` in this directory.
2. Inherit from `BaseLLMHandler` (or follow the `GoogleGenAIHandler` pattern if the provider doesn't fit the base class interface).
3. Implement `get_external_credentials()`, `get_llm()`, `get_rag_llm()`, and `generate_response()`.
4. Export the class in `__init__.py`.
