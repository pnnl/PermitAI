import os
import sys
from typing import Dict, Any, List, AsyncGenerator, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.types import ChatMessage, MessageRole
from transformers import TextGenerationPipeline
import torch
import asyncio

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_huggingface_handler.log")

logger = LogManager.get_logger(__name__)

class HuggingFaceHandler:
    """
    A handler class for interacting with HuggingFace models.

    This class provides methods to load a HuggingFace model and generate responses using it.
    """
    def __init__(self, model_path: str, adaptor_path: str = None):
        self.pipeline = self.load_model(model_path, adaptor_path)
        self.llm = PipelineLLM(self.pipeline)

    def load_model(self, model_path, adaptor_path) -> TextGenerationPipeline:
        """
        Load the HuggingFace model, tokenizer, and create the pipeline.

        Args:
            model_path (str): The path to the HuggingFace model.

        Returns:
            TextGenerationPipeline: The loaded model pipeline.

        Raises:
            Exception: If there's an error loading the model.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading model from path: {model_path} on device {device}")
            model_kwargs = dict(
                use_cache=False,
                trust_remote_code=True,
                attn_implementation="eager",  # loading the model with flash-attenstion support, flash_attention_2
                torch_dtype=torch.bfloat16,
                device_map=device
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            if adaptor_path != None:
                # load the adaptor onto the base model
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adaptor_path)
                print("***Adaptor Loaded***")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info("Model loaded successfully")
            return pipe
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the HuggingFace model.
        
        Args:
            prompt (str): The user prompt which has guidance and the actual question
        
        Returns:
            str: The generated response.
        """

        generation_args = {
            "max_new_tokens": 2000,
            "return_full_text": False,
            "temperature": 0.001,
            # "do_sample": False,
        }

        messages = [
            {"role": "user", "content": prompt}
            ]

        try:
            logger.info("Generating response")
            response = self.pipeline(messages, **generation_args)
            generated_text = response[0]['generated_text'].strip() if response else ""
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"


class PipelineLLM(LLM):
    """
    A class that wraps a HuggingFace pipeline to conform to the LLM interface.

    This class provides methods for text completion and chat functionality using a HuggingFace pipeline.
    """
    def __init__(self, pipeline: TextGenerationPipeline):
        """
        Initialize the PipelineLLM.

        Args:
            pipeline (TextGenerationPipeline): The HuggingFace pipeline to use for text generation.
        """
        super().__init__()
        self._pipeline = pipeline
        
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments for text generation.

        Returns:
            CompletionResponse: The generated completion.
        """
        default_args = {
            "max_new_tokens": 2000,
            "return_full_text": False,
            "temperature": 0.001,
            # "do_sample": False,
        }
        kwargs = {**default_args, **kwargs}
        kwargs.pop('formatted', None)
        try:
            response = self._pipeline(prompt, **kwargs)[0]['generated_text']
            new_text = response[len(prompt):].strip()
            logger.info(f"Generated completion: {new_text[:50]}...")
            return CompletionResponse(text=new_text)
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return CompletionResponse(text="Error generating response.")

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Stream the completion for the given prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments for text generation.

        Yields:
            CompletionResponse: The generated completion.
        """
        response = self.complete(prompt, **kwargs)
        yield response

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Asynchronously generate a completion for the given prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments for text generation.

        Returns:
            CompletionResponse: The generated completion.
        """
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        """
        Asynchronously stream the completion for the given prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments for text generation.

        Yields:
            CompletionResponse: The generated completion.
        """
        response = await self.acomplete(prompt, **kwargs)
        yield response

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatMessage:
        """
        Generate a chat response for the given messages.

        Args:
            messages (List[ChatMessage]): The input chat messages.
            **kwargs: Additional keyword arguments for text generation.

        Returns:
            ChatMessage: The generated chat response.
        """
        prompt = self._convert_messages_to_prompt(messages)
        response = self.complete(prompt, **kwargs)
        return ChatMessage(role=MessageRole.ASSISTANT, content=response.text)

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatMessage:
        """
        Asynchronously generate a chat response for the given messages.

        Args:
            messages (List[ChatMessage]): The input chat messages.
            **kwargs: Additional keyword arguments for text generation.

        Returns:
            ChatMessage: The generated chat response.
        """
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Generator[ChatMessage, None, None]:
        """
        Stream the chat response for the given messages.

        Args:
            messages (List[ChatMessage]): The input chat messages.
            **kwargs: Additional keyword arguments for text generation.

        Yields:
            ChatMessage: The generated chat response.
        """
        response = self.chat(messages, **kwargs)
        yield response

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatMessage, None]:
        """
        Asynchronously stream the chat response for the given messages.

        Args:
            messages (List[ChatMessage]): The input chat messages.
            **kwargs: Additional keyword arguments for text generation.

        Yields:
            ChatMessage: The generated chat response.
        """
        response = await self.achat(messages, **kwargs)
        yield response

    @property
    def metadata(self) -> LLMMetadata:
        """
        Get the metadata for the LLM.

        Returns:
            LLMMetadata: The metadata for the LLM.
        """
        return LLMMetadata()

    def _convert_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Convert a list of chat messages to a single prompt string.

        Args:
            messages (List[ChatMessage]): The input chat messages.

        Returns:
            str: The converted prompt string.
        """
        prompt = ""
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt += f"System: {message.content}\n"
            elif message.role == MessageRole.USER:
                prompt += f"Human: {message.content}\n"
            elif message.role == MessageRole.ASSISTANT:
                prompt += f"AI: {message.content}\n"
        prompt += "AI: "
        return prompt

# Example usage
if __name__ == "__main__":

    from transformers import AutoModelForCausalLM
    checkpoint_path = "/qfs/projects/policyai/models/pretrained-and-finetuned/phi3-mini-4k-instruct/nepatec1"

    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="eager",  # loading the model with flash-attenstion support, flash_attention_2
        torch_dtype=torch.bfloat16,
        device_map=None
    )
  
    nepagpt_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    for param in nepagpt_model.base_model.parameters():
        logger.info(param)
        first_tensor = param
        break

    # # load the original phi3 model
    # model_path = "microsoft/Phi-3-mini-4k-instruct"
    # original_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    # # print the first layer of the original model

    # for param in original_model.base_model.parameters():
    #     logger.info(param)
    #     second_tensor = param
    #     break

    # assert torch.allclose(first_tensor, second_tensor, atol=1e-3)
    
    
    # model_path = "/qfs/projects/policyai/models/pretrained-and-finetuned/phi3-mini-4k-instruct/nepatec1"
    # handler = HuggingFaceHandler(model_path=model_path)  # Or any other HuggingFace model
    
    # system_prompt = "You are a helpful AI assistant."
    # context = "Paris is the capital and most populous city of France."
    # question = "What is the capital of France?"

    # user_prompt = f"Read the following article and answer the question. \nContext: {context} \nQuestion: {question}\nAnswer:"
    
    
    # response = handler.generate_response(system_prompt, user_prompt)
    # logger.info(f"Response: {response}")