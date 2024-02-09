import os
import shutil
from langchain.docstore.document import Document
from transformers import GPT2TokenizerFast
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

def docs2langdoc(docs):
    """
    Converts a list of documents to language documents.

    Parameters:
    - docs (list): List of documents to convert.

    Returns:
    - list[Document]: List of language documents.
    """
    documents = []
    for doc in docs:
        if doc:
            # Read the content of the document and create a language document
            content = doc.read().decode("utf-8", errors="replace")
            source = doc.name
            lang_doc = Document(page_content=content, metadata={"source": source})
            documents.append(lang_doc)
    return documents

def split_text(documents: list[Document], chunk_size=128, chunk_overlap=20):
    """
    Splits text in documents using GPT-2 tokenizer.

    Parameters:
    - documents (list[Document]): List of documents to split.
    - chunk_size (int): Size of each text chunk (default is 128 tokens).
    - chunk_overlap (int): Number of overlapping tokens between adjacent chunks (default is 20).

    Returns:
    - list[Document]: List of split text chunks.
    """
    # Use GPT-2 tokenizer to split text in documents
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document], db_path, embedding):
    """
    Saves split text chunks to Chroma database.

    Parameters:
    - chunks (list[Document]): List of split text chunks.
    - db_path (str): Path to the Chroma database.
    - embedding: The embedding function.

    Returns:
    - None
    """
    # Clear out the database first.
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Create a new Chroma database from the split text chunks and the embedding function
    db = Chroma.from_documents(chunks, embedding, persist_directory=db_path)
    db.persist()

def chunks_2_chroma(chunks: list[Document], embedding):
    """
    Creates a Chroma database from split text chunks.

    Parameters:
    - chunks (list[Document]): List of split text chunks.
    - embedding: The embedding function.

    Returns:
    - Chroma: The Chroma database.
    """
    # Create a new Chroma database from the split text chunks and the embedding function
    db = Chroma.from_documents(chunks, embedding)
    return db