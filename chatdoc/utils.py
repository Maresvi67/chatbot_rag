import os
from chatdoc.debug import FakeChatModel, FakeTokenizer, FakeEmbeddings, FakeReranker
from langchain.chat_models.base import BaseChatModel
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from FlagEmbedding import FlagReranker

def get_llm(model: str, **kwargs) -> BaseChatModel:
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

def get_tokenizer(model: str, **kwargs) -> AutoTokenizer:
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

def get_embedding(model: str, **kwargs) -> HuggingFaceEmbeddings:
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

def get_reranker(model: str, **kwargs) -> FlagReranker:
    """
    Get a reranker instance based on the specified model.

    Parameters:
    - model (str): The name of the reranker model.

    Returns:
    - FlagReranker: An instance of the reranker model.
    """
    if model == "debug":
        return FakeReranker()

    if "bge-reranker" in model:
        reranker = FlagReranker(model)
        return reranker

    raise NotImplementedError(f"Reranker {model} not supported!")

def get_db(chroma_path="D:\Chroma", embedding=None, **kwargs) -> Chroma:
    """
    Get a Chroma vector store instance.

    Parameters:
    - chroma_path (str): The path to the Chroma vector store.
    - embedding: The embedding function to use.

    Returns:
    - Chroma: An instance of the Chroma vector store or None if the path doesn't exist.
    """
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding)
    else:
        db = None
    return db
