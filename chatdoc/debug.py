from langchain.chat_models.fake import FakeListChatModel


class FakeChatModel(FakeListChatModel):
    def __init__(self, **kwargs):
        responses = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit"]
        super().__init__(responses=responses, **kwargs)

    def text_generation(self, query, **kwargs):
        return("Lorem ipsum dolor sit amet, consectetur adipiscing elit")

class FakeTokenizer():
    def __init__(self):
        pass

    def apply_chat_template(self, msgs_list, **kwargs):
        return("\n".join([str(msg) for msg in msgs_list]))