import os
import streamlit as st

from dotenv import load_dotenv
from chatdoc.chatbot import LLMChatBot
from chatdoc.chat_message_histories import CustomChatHistory
from chatdoc.utils import get_llm, get_tokenizer, get_emdedding, get_db
from chatdoc.vector_store import docs2langdoc, split_text, save_to_chroma

# Load enviroment variables
load_dotenv()

# Parameters
LLM_MODEL_NAME = "debug"
EMBEDDING_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PATH="D:\Chroma"

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

# Set DB
chroma_db = get_db(CHROMA_PATH, embedding=embedding)

# Set LLMChatBot
llm_chatbot = LLMChatBot(llm=llm_client, tokenizer=tokenizer, embedding=embedding, chroma_db=chroma_db)

# Convert data to chroma
uploaded_files = st.sidebar.file_uploader(
    "Upload .md files",
    type=["md"],
    help="Only Markdown files are supported",
    accept_multiple_files=True)

if uploaded_files:

    del chroma_db
    
    with st.spinner("Indexing document... This may take a while⏳"):
        documents = docs2langdoc(uploaded_files)
        chunks = split_text(documents)
        save_to_chroma(chunks, CHROMA_PATH, embedding)

    chroma_db = get_db(CHROMA_PATH, embedding=embedding)

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
