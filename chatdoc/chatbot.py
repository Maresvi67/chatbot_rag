from chatdoc.chat_message_histories import CustomChatHistory


class LLMChatBot():
    def __init__(self, llm):
        """
        Initializes the Language Model Chat Bot.

        Parameters:
        - None

        Returns:
        - None
        """
        self.llm = llm


    def generate_expanded_query(self, query:str, n_query: int=5):
        """
        Generates an expanded query by repeating the input query.

        This method is based on https://arxiv.org/abs/2305.03653.

        Parameters:
        - query (str): The input query.
        - n_query (int): The number of times the input query will be repeated to expand the query.

        Returns:
        - str: The expanded query.
        """
        
        llm_output = self.llm.text_generation(query)

        expanded_query = "\n".join(n_query * [query]) + "\n" + llm_output
        
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
