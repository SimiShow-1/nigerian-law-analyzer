import os
import streamlit as st
from lexa_core import LexaCore
from dotenv import load_dotenv

load_dotenv()

# Page setup
st.set_page_config(page_title="Lexa – Nigerian Legal Assistant", layout="wide")

# CSS
st.markdown("""
    <style>
    .chat-container { max-width: 700px; margin: auto; padding: 10px; }
    .chat-bubble { padding: 10px 14px; border-radius: 14px; line-height: 1.4; }
    .user { background: #dcf8c6; text-align: right; }
    .assistant { background: #fff; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

# --- Header Logo ---
st.image("lexa_logo.png", width=150)

# Initialize backend
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Init error: {e}")
        st.stop()

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Clear button
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Chat display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        cols = st.columns([0.2, 0.8])
        with cols[1]:
            st.markdown(f'<div class="chat-bubble user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        cols = st.columns([0.1, 0.9])
        with cols[0]:
            st.image("lexa_logo.png", width=32)
        with cols[1]:
            st.markdown(f'<div class="chat-bubble assistant">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask Lexa a legal question..."):
    # append user
    st.session_state.messages.append({"role": "user", "content": prompt})
    # render immediately
    cols = st.columns([0.2, 0.8])
    with cols[1]:
        st.markdown(f'<div class="chat-bubble user">{prompt}</div>', unsafe_allow_html=True)

    # get response
    with st.spinner("Lexa is thinking..."):
        try:
            reply = st.session_state.lexa.process_query(prompt)
        except Exception as e:
            reply = "Sorry, I encountered an error. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    cols = st.columns([0.1, 0.9])
    with cols[0]:
        st.image("lexa_logo.png", width=32)
    with cols[1]:
        st.markdown(f'<div class="chat-bubble assistant">{reply}</div>', unsafe_allow_html=True)
