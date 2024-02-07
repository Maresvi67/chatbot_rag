from langchain.chat_models.fake import FakeListChatModel


class FakeChatModel(FakeListChatModel):
    def __init__(self, **kwargs):
        responses = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit"]
        super().__init__(responses=responses, **kwargs)

    def text_generation(self, query):
        return("Lorem ipsum dolor sit amet, consectetur adipiscing elit")