from chatdoc.chat_message_histories import CustomChatHistory

class LLMChatBot():
    def __init__(self):
        """
        Initializes the Language Model Chat Bot.

        Parameters:
        - None

        Returns:
        - None
        """

    def answer(self, messages : CustomChatHistory):
        """
        Generates an answer based on the provided chat message history.

        Parameters:
        - messages (CustomChatHistory): The chat message history.

        Returns:
        - str: The generated answer.
        """
        generated_answer = "I\'ll be ready to answer your questions shortly!"
        return generated_answer
