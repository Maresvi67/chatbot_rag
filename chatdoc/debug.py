from langchain.chat_models.fake import FakeListChatModel
from langchain.embeddings.fake import FakeEmbeddings as FakeEmbeddingsBase


class FakeChatModel(FakeListChatModel):
    def __init__(self, **kwargs):
        responses = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit"]
        super().__init__(responses=responses, **kwargs)

    def text_generation(self, query, **kwargs):
        return("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
        

class FakeEmbeddings(FakeEmbeddingsBase):
    def __init__(self, **kwargs):
        super().__init__(size=4, **kwargs)

class FakeReranker():
    def __init__(self, **kwargs):
        pass

class FakeTokenizer():
    def __init__(self):
        pass

    def apply_chat_template(self, msgs_list, **kwargs):
        return("\n".join([str(msg) for msg in msgs_list]))