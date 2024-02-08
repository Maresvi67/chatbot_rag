import os
import streamlit as st

from dotenv import load_dotenv
from chatdoc.chatbot import LLMChatBot
from chatdoc.chat_message_histories import CustomChatHistory
from chatdoc.utils import get_llm, get_tokenizer, get_emdedding, get_db, get_reranker
from chatdoc.vector_store import docs2langdoc, split_text, chunks_2_chroma
from streamlit import session_state as ss

# Load enviroment variables
load_dotenv()

# Parameters
LLM_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EMBEDDING_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = 'BAAI/bge-reranker-large'

# Set title of the page
st.set_page_config(page_title="Company X: Chat with Documents", page_icon="🤖")
st.title("🤖 Company X: Chat with Documents")

# Setup memory for contextual conversation
msgs = CustomChatHistory()

# Set LLM
if 'llm_client' not in ss:
    ss['llm_client'] = get_llm(LLM_MODEL_NAME)

llm_client = ss['llm_client']

# Set Embedding
if 'embedding' not in ss:
    ss['embedding'] = get_emdedding(EMBEDDING_NAME)

embedding = ss['embedding']

# Set Tokenizer
if 'tokenizer' not in ss:
    ss['tokenizer'] = get_tokenizer(LLM_MODEL_NAME)
    
tokenizer = ss['tokenizer']

# Set Reranker
if 'reranker' not in ss:
    ss['reranker'] = get_reranker(RERANKER_NAME)
    
reranker = ss['reranker']

# Convert data to chroma
if 'show_uploader' not in ss:
    ss['show_uploader'] = True

if 'chroma' not in ss:
    ss['chroma'] = None

if ss['show_uploader']:
    uploaded_files = st.sidebar.file_uploader(
        "Upload .md files",
        type=["md"],
        help="Only Markdown files are supported",
        accept_multiple_files=True)

    if len(uploaded_files) > 0:
        with st.spinner("Indexing document... This may take a while⏳"):
            documents = docs2langdoc(uploaded_files)
            chunks = split_text(documents)
            chroma_db = chunks_2_chroma(chunks, embedding)
        ss['chroma'] = chroma_db
        ss['show_uploader'] = False

if ss['chroma'] is not None:
    chroma_db = ss['chroma']
else:
    st.stop()

# Set LLMChatBot
llm_chatbot = LLMChatBot(llm=llm_client, tokenizer=tokenizer, embedding=embedding, reranker=reranker, chroma_db=chroma_db)

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
