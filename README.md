# Chatbot_RAG

A FAQ chatbot for connecting your external data sources to an LLM using the RAG (Retrieval-Augmented Generation) technique, combining HuggingFace, LangChain, and Streamlit.

## Overview

This project implements a chatbot that leverages the RAG technique to provide answers based on external data sources. It connects HuggingFace models, LangChain functionalities, and a user-friendly interface using Streamlit.

## Note on Python Version
This project requires Python 3.11.7. Make sure to use this version to ensure compatibility with all dependencies.

## Running

To run the chatbot, follow these steps:

```shell
# Clone the repository
$ git clone <repository_url>
$ cd chatbot_rag

# Install dependencies
$ pip install -r requirements.txt

# Run the main application
$ python -m streamlit run main.py
```
Visit [http://localhost:8501](http://localhost:8501) in your web browser to interact with the chatbot through the Streamlit interface.

## Features
* Language Model: Integration of HuggingFace language models.
* Tokenizer: Utilizes GPT-2 tokenizer for text processing.
* Embedding: Incorporates LangChain embeddings for enhanced performance.
* Document Retrieval: Uses ChromaDB for efficient document retrieval.
* Reranking: Applies a reranker for refining document ranking.
* User Interface: Streamlit-based user interface for a seamless experience.

## How It Works
The chatbot follows the Retrieval-Augmented Generation (RAG) technique, where the initial query is expanded, relevant documents are retrieved using ChromaDB, and a language model generates responses based on the retrieved documents.

## Usage
1. Launch the application using the provided commands.
2. Interact with the chatbot by typing your queries in the provided input field.
3. Receive answers generated based on the combined power of RAG, LangChain, and HuggingFace.

The first excution could take a time for download several component from HuggingFace.

## References
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks ([link](https://arxiv.org/abs/2005.11401))
Mistral 7B ([link](https://arxiv.org/abs/2310.06825))