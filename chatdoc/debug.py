from langchain.chat_models.fake import FakeListChatModel
from langchain.embeddings.fake import FakeEmbeddings as FakeEmbeddingsBase


class FakeChatModel(FakeListChatModel):
    """
    A fake chat model for testing and debugging purposes.

    This model generates predefined responses and can be used as a placeholder for a chat-based language model.

    Attributes:
    - responses (list): Predefined list of responses for generating outputs.
    """

    def __init__(self, **kwargs):
        responses = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit"]
        super().__init__(responses=responses, **kwargs)

    def text_generation(self, query, **kwargs):
        """
        Generate a fake response for a given query.

        Parameters:
        - query (str): The input query for generating a response.

        Returns:
        - str: The generated fake response.
        """
        return "Lorem ipsum dolor sit amet, consectetur adipiscing elit"


class FakeEmbeddings(FakeEmbeddingsBase):
    """
    Fake embeddings class for testing and debugging purposes.

    This class provides fake embeddings with a specified size for use in fake chat models.
    """

    def __init__(self, **kwargs):
        super().__init__(size=4, **kwargs)


class FakeReranker:
    """
    Fake reranker class for testing and debugging purposes.

    This class is a placeholder for a reranker and does not perform any actual reranking.
    """

    def __init__(self, **kwargs):
        pass


class FakeTokenizer:
    """
    Fake tokenizer class for testing and debugging purposes.

    This class provides a simple fake tokenizer that joins messages in a list into a newline-separated string.
    """

    def __init__(self):
        pass

    def apply_chat_template(self, msgs_list, **kwargs):
        """
        Apply a fake chat template to a list of messages.

        Parameters:
        - msgs_list (list): List of messages to be processed.

        Returns:
        - str: Joined messages as a newline-separated string.
        """
        return "\n".join([str(msg) for msg in msgs_list])
