from chatdoc.chat_message_histories import CustomChatHistory
from chatdoc.prompts import prompt_q2d_zs, prompt_cot, prompt_conv


class LLMChatBot():
    def __init__(self, llm, tokenizer):
        """
        Initializes the Language Model Chat Bot.

        Parameters:
        - None

        Returns:
        - None
        """
        self.llm = llm
        self.tokenizer = tokenizer


    def generate_expanded_query(self, query:str, n_query: int=5, max_new_tokens=512):
        """
        Generates an expanded query by repeating the input query.

        This method is based on https://arxiv.org/abs/2305.03653.

        Parameters:
        - query (str): The input query.
        - n_query (int): The number of times the input query will be repeated to expand the query.

        Returns:
        - str: The expanded query.
        """
        tokenize_query = self.tokenizer.apply_chat_template([{"role" : "user", "content" : query}], tokenize=False)
        prompt_query = prompt_cot.format(query=tokenize_query)
        llm_output = self.llm.text_generation(prompt_query, max_new_tokens=max_new_tokens)

        expanded_query = "\n\n".join(n_query * [query]) + "\n\n" + llm_output
        
        return expanded_query

    def answer(self, messages : CustomChatHistory):
        """
        Generates an answer based on the provided chat message history.

        Parameters:
        - messages (CustomChatHistory): The chat message history.

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

        # Retriver

        # LLM

        generated_answer = expanded_query
        return generated_answer
