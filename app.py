import os
import streamlit as st
from lexa_core import LexaCore
from dotenv import load_dotenv

load_dotenv()

# Page setup
st.set_page_config(page_title="Lexa – Nigerian Legal Assistant", layout="wide")

# CSS for bubbles
st.markdown("""
    <style>
    .chat-container { max-width: 700px; margin: auto; padding: 10px; background: #fafafa; border-radius: 10px; }
    .chat-bubble { padding: 10px 14px; border-radius: 14px; line-height: 1.4; color: #000; }
    .user { background: #dcf8c6; text-align: right; }
    .assistant { background: #fff; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# --- Header Logo via st.image ---
st.image("lexa_logo.png", width=150)

# Initialize backend
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

# Init or clear messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Render chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    content = msg["content"].replace("\n", "<br>")
    if msg["role"] == "user":
        # User bubble (no avatar)
        st.markdown(f'<div class="chat-bubble user">{content}</div>', unsafe_allow_html=True)
    else:
        # Assistant with avatar
        cols = st.columns([0.1, 0.9])
        with cols[0]:
            st.image("lexa_logo.png", width=32)
        with cols[1]:
            st.markdown(f'<div class="chat-bubble assistant">{content}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input prompt
if prompt := st.chat_input("Ask Lexa a legal question..."):
    # Append and render user
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble user">{prompt}</div>', unsafe_allow_html=True)

    # Get and render assistant reply
    with st.spinner("Lexa is thinking..."):
        try:
            reply = st.session_state.lexa.process_query(prompt)
        except Exception:
            reply = "Sorry, I encountered an error. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    cols = st.columns([0.1, 0.9])
    with cols[0]:
        st.image("lexa_logo.png", width=32)
    with cols[1]:
        st.markdown(f'<div class="chat-bubble assistant">{reply}</div>', unsafe_allow_html=True)
