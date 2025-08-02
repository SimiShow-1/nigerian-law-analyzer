import streamlit as st
from streamlit_chat import message
from lexa_core import LexaCore
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lexa - Nigerian Legal Assistant", layout="wide")

# Load CSS
if os.path.exists("style.css"):
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    logger.warning("No style.css found")

# Logo at the top (larger)
st.image("lexa_logo.png", width=120)

# Session state
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Failed to initialize Lexa: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Clear chat button
st.button("🧹 Clear Chat", on_click=lambda: st.session_state.update({
    "messages": [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]
}))

# Display all messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        # Display Lexa's logo manually
        cols = st.columns([0.1, 0.9])
        with cols[0]:
            st.image("lexa_logo.png", width=40)
        with cols[1]:
            message(msg["content"], key=f"assistant_{i}")

# Chat input
if prompt := st.chat_input("Ask a legal question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = st.session_state.lexa.process_query(prompt)
    except Exception as e:
        response = "Sorry, something went wrong. Please try again later."
        logger.error(e)

    st.session_state.messages.append({"role": "assistant", "content": response})
