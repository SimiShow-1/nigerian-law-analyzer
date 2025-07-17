import streamlit as st
from streamlit_chat import message
import os
from lexa_core import LexaCore, LexaError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'lexa' not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except LexaError as e:
        st.error(f"❌ Failed to initialize Lexa: {str(e)}")
        st.stop()

# Page Config
st.set_page_config(page_title="Lexa - Nigerian Legal AI", page_icon="⚖️", layout="wide")

# Custom CSS for WhatsApp-style UI
st.markdown("""
<style>
.chat-container {
    height: 70vh;
    overflow-y: auto;
    padding-bottom: 80px;
}
.input-container {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: white;
    padding: 10px;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding-right: 50px;
}
button.send-button {
    position: absolute;
    right: 20px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Logo + Title
st.image("lexa_logo.png", width=100)
st.markdown("## **Lexa** – Your Nigerian Legal AI Assistant")

# Chat history UI
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        message(msg['content'], is_user=(msg['role'] == 'user'), key=f"{msg['role']}_{i}")
    st.markdown('</div>', unsafe_allow_html=True)

# Input area
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([11, 1])
    with col1:
        user_input = st.text_input("Ask a legal question...", key="input", label_visibility="collapsed")
    with col2:
        if st.button("✈️", key="send", use_container_width=True):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
                try:
                    with st.spinner("Lexa is thinking..."):
                        reply = st.session_state.lexa.process_query(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()

                except LexaError as e:
                    st.error(f"⚠️ Lexa Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
