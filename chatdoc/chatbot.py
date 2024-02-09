from chatdoc.chat_message_histories import CustomChatHistory
from chatdoc.prompts import prompt_q2d_zs, prompt_cot, prompt_conv, prompt_cot_prf, prompt_cot_prf_mod

class LLMChatBot():
    """
    Language Model Chat Bot that uses a combination of document retrieval, reranking, and a language model to generate answers.

    Attributes:
    - llm: The language model instance.
    - tokenizer: The tokenizer used for processing text.
    - embedding: The embedding model.
    - reranker: The reranker instance for document reranking.
    - chroma_db: The ChromaDB instance for document retrieval.
    """
    def __init__(self, llm, tokenizer, embedding, reranker, chroma_db):
        """
        Initializes the Language Model Chat Bot. 

        Parameters:
        - llm: The language model instance.
        - tokenizer: The tokenizer used for processing text.
        - embedding: The embedding model.
        - reranker: The reranker instance for document reranking.
        - chroma_db: The ChromaDB instance for document retrieval.
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.chroma_db = chroma_db
        self.reranker = reranker

    def generate_expanded_query(self, query:str, n_query: int=5, max_new_tokens=512):
        """
        Generates an expanded query by repeating the input query.

        This method is based on https://arxiv.org/abs/2305.03653.

        Parameters:
        - query (str): The input query.
        - n_query (int): The number of times the input query will be repeated to expand the query.
        - max_new_tokens (int): The maximum number of tokens allowed in the generated response.

        Returns:
        - str: The expanded query.
        """
        # Tokenize the query and generate an expanded query using language model
        tokenize_query = self.tokenizer.apply_chat_template([{"role" : "user", "content" : query}], tokenize=False)
        prompt_query = prompt_cot.format(query=tokenize_query)
        llm_output = self.llm.text_generation(prompt_query, max_new_tokens=max_new_tokens)

        expanded_query = "\n\n".join(n_query * [query]) + "\n\n" + llm_output
        return expanded_query

    def retrieve_documents(self, query:str, k: int=5):
        """
        Retrieves relevant documents based on the input query.

        Parameters:
        - query (str): The input query.
        - k (int): The number of documents to retrieve.

        Returns:
        - list: Relevant documents.
        """
        # Use ChromaDB retriever to get relevant documents
        retriever = self.chroma_db.as_retriever(search_kwargs={"k": k})
        retieval_docs = retriever.get_relevant_documents(query)
        return retieval_docs

    def rerank_docs(self, query, retieval_docs, th=0):
        """
        Reranks retrieved documents based on the input query and reranker instance.

        Parameters:
        - query (str): The input query.
        - retieval_docs (list): The list of retrieved documents.
        - th (float): Reranking threshold.

        Returns:
        - list: Reranked documents.
        """
        # Rerank the documents based on the query and threshold
        query_and_docs  = [[query, i.page_content] for i in retieval_docs]
        scores = self.reranker.compute_score(query_and_docs)
        sorted_docs = sorted(zip(scores, retieval_docs), reverse=True)
        reranker_docs = [i[1] for i in sorted_docs if i[0] > th]
        return reranker_docs

    def answer(self, messages : CustomChatHistory, max_new_tokens=512):
        """
        Generates an answer based on the provided chat message history.

        Parameters:
        - messages (CustomChatHistory): The chat message history.
        - max_new_tokens (int): The maximum number of tokens allowed in the generated response.

        Returns:
        - str: The generated answer.
        """
        # Convert messages to list
        msgs_list = messages.convert_to_list()

        # Extract last user message
        last_message = msgs_list[-1]

        # Validate if the role of the last massage is user
        if last_message["role"] != "user":
            raise NotImplementedError("Only user messages are supported for generating answers.")

        # Extract query from last user message
        query = last_message["content"]

        # Expand query
        expanded_query = self.generate_expanded_query(query)

        # Retriever
        retieval_docs = self.retrieve_documents(expanded_query)

        # Reranker
        reranked_docs = self.rerank_docs(query, retieval_docs)

        # LLM
        if len(reranked_docs) > 0:
            # If reranked documents are available, create a context for the prompt
            context_text = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
            prompt_query = prompt_cot_prf_mod.format(query=query, context=context_text)
        else:
            # If no reranked documents, use a basic prompt
            prompt_query = prompt_cot.format(query=query)

        # Tokenize the query and generate the answer using the language model
        tokenize_query = self.tokenizer.apply_chat_template([{"role" : "user", "content" : prompt_query}], tokenize=False, add_generation_prompt=True)
        generated_answer = self.llm.text_generation(prompt_query, max_new_tokens=max_new_tokens)

        if len(reranked_docs) > 0:
            # If reranked documents are available, include them in the result
            docs_output = "\n\n".join([doc.metadata["source"] for doc in reranked_docs])
            result = f"""{generated_answer}\n\nThis answer is derived from the information found in the following documents:\n\n{docs_output}"""
        else:
            result = f"""{generated_answer}\n\nNo files directly address this question; however, the answer provided may offer valuable insights."""

        return result
