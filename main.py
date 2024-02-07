import os
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Set title of the page
st.set_page_config(page_title="Company X: Chat with Documents", page_icon="🤖")
st.title("🤖 Company X: Chat with Documents")

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

# Clear messagge
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

# Generete the messages in the interface
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Take the user input
if user_query := st.chat_input(placeholder="Ask me anything!"):
    # Add user input to chat history
    msgs.add_user_message(user_query)

    # Add user input to chat interface
    with st.chat_message("user"):
        st.write(user_query)

    # Generate the result of the chatbot
    result = "I\'ll be ready to answer your questions shortly!"

    # Add assistant input to chat history
    msgs.add_ai_message(result)

    # Add assistant input to chat interface
    with st.chat_message("assistant"):
        st.write(result)
