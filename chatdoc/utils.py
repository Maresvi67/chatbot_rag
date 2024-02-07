from chatdoc.debug import FakeChatModel, FakeTokenizer
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

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

# prompt__ = "This is a conversation, Always generate grammatically correct sentences that no repeat. The user is a very understanding. Respond to recent discussion between user and you"
