from chatdoc.debug import FakeChatModel, FakeTokenizer, FakeEmbeddings
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

def get_llm(model: str, **kwargs):
    if model == "debug":
        return FakeChatModel()

    if "mistral" in model:
        return InferenceClient(model)

    raise NotImplementedError(f"Model {model} not supported!")

def get_tokenizer(model: str, **kwargs):
    if model == "debug":
        return FakeTokenizer()

    if "mistral" in model:
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    raise NotImplementedError(f"Tokenizer {model} not supported!")

def get_emdedding(model: str, **kwargs):
    if model == "debug":
        return FakeEmbeddings()

    if "bge-small" in model:
        embedding_hf = SentenceTransformer(model)
        return embedding_hf

    raise NotImplementedError(f"Embedding {model} not supported!")