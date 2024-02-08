import os
from chatdoc.debug import FakeChatModel, FakeTokenizer, FakeEmbeddings
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

def get_llm(model: str, **kwargs):
    """
    Get a Language Model instance based on the specified model.

    Parameters:
    - model (str): The name of the language model.

    Returns:
    - BaseChatModel: An instance of the language model.
    """
    if model == "debug":
        return FakeChatModel()

    if "mistral" in model:
        return InferenceClient(model)

    raise NotImplementedError(f"Model {model} not supported!")

def get_tokenizer(model: str, **kwargs):
    """
    Get a tokenizer instance based on the specified model.

    Parameters:
    - model (str): The name of the tokenizer.

    Returns:
    - AutoTokenizer: An instance of the tokenizer.
    """
    if model == "debug":
        return FakeTokenizer()

    if "mistral" in model:
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    raise NotImplementedError(f"Tokenizer {model} not supported!")

def get_emdedding(model: str, **kwargs):
    """
    Get an embedding instance based on the specified model.

    Parameters:
    - model (str): The name of the embedding model.

    Returns:
    - HuggingFaceEmbeddings: An instance of the embedding model.
    """
    if model == "debug":
        return FakeEmbeddings()

    if "bge-small" in model:
        embedding_hf = HuggingFaceEmbeddings(model_name=model)
        return embedding_hf

    raise NotImplementedError(f"Embedding {model} not supported!")

def get_db(chome_path="D:\Chroma", embedding=None, **kwargs):
    """
    Get a Chroma vector store instance.

    Parameters:
    - chroma_path (str): The path to the Chroma vector store.
    - embedding: The embedding function to use.

    Returns:
    - Chroma: An instance of the Chroma vector store.
    """
    if os.path.exists(chome_path) and os.path.isdir(chome_path):
        db = Chroma(persist_directory=chome_path, embedding_function=embedding)
    else:
        db = None
    return db