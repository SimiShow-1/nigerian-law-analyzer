
import streamlit as st
from streamlit_chat import message
from lexa_core import LexaCore
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lexa - Nigerian Legal Assistant", layout="wide")

# Load custom CSS
css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    logger.warning("style.css not found, using default styles")

# Initialize session state
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Failed to initialize Lexa: {e}")
        st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}]

# Header
st.markdown("<h1>Lexa: Nigerian Legal Assistant</h1>", unsafe_allow_html=True)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}]

# Display chat messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], avatar_style="lexa-logo", key=f"assistant_{i}")

# Chat input at bottom
if prompt := st.chat_input("Ask a legal question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = st.session_state.lexa.process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error processing query: {e}")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
