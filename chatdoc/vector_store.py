import os
import shutil
from langchain.docstore.document import Document
from transformers import GPT2TokenizerFast
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores.chroma import Chroma

def docs2langdoc(docs):
    documents = []
    for doc in docs:
        if doc:
            content = doc.read().decode("utf-8", errors="replace")
            source = doc.name

            lang_doc = Document(page_content=content, metadata={"source": source})

            documents.append(lang_doc)
    return documents

def split_text(documents: list[Document]):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = MarkdownTextSplitter.from_huggingface_tokenizer(tokenizer,
                                                                                chunk_size=512,
                                                                                chunk_overlap=20
                                                                                )

    chunks = text_splitter.split_documents(documents)
    return chunks

def save_to_chroma(chunks: list[Document], db_path, embedding):
    # Clear out the database first.
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding, persist_directory=db_path
    )
    db.persist()