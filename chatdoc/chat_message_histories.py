from langchain_community.chat_message_histories import StreamlitChatMessageHistory

class CustomChatHistory(StreamlitChatMessageHistory):
    """
    Customized chat message history class for Streamlit applications.

    This class extends StreamlitChatMessageHistory and includes additional functionality.

    Attributes:
    - avatars (dict): A dictionary mapping message types to roles (e.g., {"human": "user", "ai": "assistant"}).
    """

    def __init__(self):
        """
        Initialize the CustomChatHistory instance.

        Sets up the avatars dictionary to map message types to roles.
        """
        super().__init__()
        
        self.avatars = {"human": "user", "ai": "assistant"}

    def convert_to_list(self, init_index=1):
        """
        Convert messages attributes to a list.

        Parameters:
        - init_index (int): The starting index for extracting messages.

        Returns: 
        - list: A list containing dictionaries with message attributes.
        """
        messages_list = []

        for message in self.messages[init_index:]:
            message_dict = {
                "role": self.avatars[message.type],
                "content": message.content
            }
            messages_list.append(message_dict)

        return messages_list