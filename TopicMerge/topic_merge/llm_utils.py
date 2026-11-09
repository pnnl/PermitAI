import dspy
import logging
from google import genai

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('topic_merge')


_client_cache = {}

def get_client(api_key):
    """Return a cached Google AI Studio GenAI client."""
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(api_key=api_key)
    return _client_cache[api_key]


def get_lm(model="gemini-2.5-flash", api_key=None, temperature=1.0, **kwargs):
    if api_key is None:
        raise ValueError("api_key cannot be None.")
    return dspy.LM(model=f"gemini/{model}", api_key=api_key, temperature=temperature, **kwargs)


class Extractor(dspy.Module):
    """Thin DSPy wrapper that retries extraction up to 2 times on failure."""
    def __init__(self, signature, reasoning_head, metadata_name):
        super().__init__()
        self.extractor = dspy.Predict(signature) if reasoning_head != 'cot' else dspy.ChainOfThought(signature)
        self.metadata_name = metadata_name

    def forward(self, **kwargs):
        for i in range(2):
            try:
                return self.extractor(**kwargs)
            except Exception as e:
                print(f"Extraction error: {e}. Retrying ({i+1}/2)...")
