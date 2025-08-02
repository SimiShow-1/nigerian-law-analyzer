
import streamlit as st
from streamlit_chat import message
from lexa_core import LexaCore
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lexa - Nigerian Legal Assistant", layout="wide")

# Load CSS
css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    logger.warning("style.css not found")

# Logo
st.image("lexa_logo.png", width=150)

# Init session state
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Failed to initialize Lexa: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"
        }
    ]

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"
        }
    ]

# Display chat
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    message(
        msg["content"],
        is_user=is_user,
        key=f"{msg['role']}_{i}",
        avatar_style="url" if not is_user else None,
        avatar_url="lexa_logo.png" if not is_user else None
    )

# Chat input
if prompt := st.chat_input("Ask a legal question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Lexa is thinking..."):
        try:
            response = st.session_state.lexa.process_query(prompt)
        except Exception as e:
            response = "Sorry, I encountered an error. Please try again."
            st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()  # Force immediate refresh
