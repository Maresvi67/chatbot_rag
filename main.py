import os
import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Set title of the page
st.set_page_config(page_title="Company X: Chat with Documents", page_icon="🤖")
st.title("🤖 Company X: Chat with Documents")

# Clear messagge
st.sidebar.button("Clear message history")

# First assitant message
st.chat_message("assistant").write("How can I help you?")

# Take the user input
if user_query := st.chat_input(placeholder="Ask me anything!"):

    # Add user input to chat interface
    with st.chat_message("user"):
        st.write(user_query)
