from langchain_community.chat_message_histories import StreamlitChatMessageHistory

class CustomChatHistory(StreamlitChatMessageHistory):
    def __init__(self):
        super().__init__()
        
        self.avatars = {"human": "user", "ai": "assistant"}


    def convert_to_list(self, init_index=1):
        """
        Convert messages attributes to a list.

        Returns:
        list: A list containing dictionaries with message attributes.
        """
        messages_list = []

        for message in self.messages[init_index:]:
            message_dict = {
                "role": self.avatars[message.type],
                "content": message.content
            }
            messages_list.append(message_dict)

        return messages_list