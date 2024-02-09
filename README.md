# Chatbot_RAG

A FAQ chatbot for connecting your external data sources to an LLM using the RAG (Retrieval-Augmented Generation) technique, combining HuggingFace, LangChain, and Streamlit.

## Overview

This project implements a chatbot that leverages the RAG technique to provide answers based on external data sources. It connects HuggingFace models, LangChain functionalities, and a user-friendly interface using Streamlit.

## Note on Python Version
This project requires Python 3.11.7. Make sure to use this version to ensure compatibility with all dependencies.

## Running

To run the chatbot, follow these steps:
1. Clone the repository
```shell
$ git clone https://github.com/cocol428/chatbot_rag.git
$ cd chatbot_rag
```
2. Create a virtual environment (optional but recommended)
```shell
$ python -m venv venv
$ source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install dependencies
```shell
$ pip install -r requirements.txt
```
4. Create an .env file in the project root and set the HUGGINGFACEHUB_API_TOKEN:
```shell
HUGGINGFACEHUB_API_TOKEN=your_token_here
```
5. Run the main application
```shell
$ python -m streamlit run main.py
```
Visit [http://localhost:8501](http://localhost:8501) in your web browser to interact with the chatbot through the Streamlit interface.

## Features
* Language Model: Integration of HuggingFace language models. ([mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1))
* Tokenizer: Utilizes GPT-2 tokenizer for text processing. ([mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1))
* Embedding: Incorporates LangChain embeddings for enhanced performance. ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5))
* Document Retrieval: Uses ChromaDB for efficient document retrieval.
* Reranking: Applies a reranker for refining document ranking. ([BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large))
* User Interface: Streamlit-based user interface for a seamless experience.

## How It Works

The chatbot operates using the Retrieval-Augmented Generation (RAG) technique, involving the following steps:

1. **Vector Base Generation:**
   - The Vector Base is generated with ChromaDB and an Embedding.
2. **Query Expansion:**
   - The initial query undergoes expansion by concatenating it four times with the response of a Language Model (LLM) using a specific prompt. Detailed information on this process can be found in the first article mentioned in the references section.
3. **Document Retrieval:**
   - Relevant documents are retrieved using ChromaDB. This involves obtaining the top k (default 5) chunks of documents most similar to the query.
4. **Reranking:**
   - A reranker is employed to precisely assess the relevance of each retrieved document with respect to a given query. The system applies a default threshold of 0.5; if the relevance score falls below this threshold, the document will not be considered in the retrieval results. If no documents meet the threshold criteria, the final result will be generated using a zero-shot approach, without leveraging context from any retrieved documents.
5. **Response Generation:**
   - A language model generates responses based on the retrieved documents using prompt engineering. Several of these prompts are based on the reported prompt in the first article.

## How to Use
1. Start by launching the application with the provided commands.
2. Load your Markdown files into memory using the left-side uploader. You can select multiple files, but keep in mind that vectorization may take some time.
3. Interact with the chatbot by entering your queries in the provided input field.
4. You have the option to clear the chat history, but note that this won't delete the vector database for context documents.
5. If you want to initiate a new conversation with different files, you'll need to reset the application by running it again.

**The initial run may require some time as it involves downloading various components from HuggingFace.**

## References
1. Query Expansion by Prompting Large Language Models ([link](https://arxiv.org/abs/2305.03653))
2. Mistral 7B ([link](https://arxiv.org/abs/2310.06825))
