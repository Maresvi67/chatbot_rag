import os
import streamlit as st

from dotenv import load_dotenv
from chatdoc.chatbot import LLMChatBot
from chatdoc.chat_message_histories import CustomChatHistory
from chatdoc.utils import get_llm, get_tokenizer, get_emdedding

# Load enviroment variables
load_dotenv()

# Parameters
LLM_MODEL_NAME = "debug"

# Set title of the page
st.set_page_config(page_title="Company X: Chat with Documents", page_icon="🤖")
st.title("🤖 Company X: Chat with Documents")

# Setup memory for contextual conversation
msgs = CustomChatHistory()

# Set LLM
llm_client = get_llm(LLM_MODEL_NAME)

# Set Embedding
embedding = get_emdedding(EMBEDDING_NAME)

# Set Tokenizer
tokenizer = get_tokenizer(LLM_MODEL_NAME)

# Set LLMChatBot
llm_chatbot = LLMChatBot(llm=llm_client, tokenizer=tokenizer)

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
    result = llm_chatbot.answer(msgs)

    # Add assistant input to chat history
    msgs.add_ai_message(result)

    # Add assistant input to chat interface
    with st.chat_message("assistant"):
        st.write(result)
