# app.py
import os
import streamlit as st
from lexa_core import LexaCore
from dotenv import load_dotenv

load_dotenv()

# --- Page configuration ---
st.set_page_config(page_title="Lexa – Nigerian Legal Assistant", layout="wide")

# --- Styles for custom chat bubbles and layout ---
st.markdown("""
    <style>
    .header-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .header-logo img {
        width: 150px;  /* slightly larger main logo */
        border-radius: 8px;
    }
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 10px;
        background: #fafafa;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .chat-message.user {
        justify-content: flex-end;
    }
    .chat-bubble {
        max-width: 80%;
        padding: 10px 14px;
        border-radius: 14px;
        line-height: 1.4;
    }
    .chat-bubble.user {
        background: #dcf8c6;
        color: #000;
        border-bottom-right-radius: 4px;
    }
    .chat-bubble.assistant {
        background: #ffffff;
        color: #000;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
    }
    .chat-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .chat-message.user .chat-avatar {
        display: none;  /* no avatar for user */
    }
    </style>
""", unsafe_allow_html=True)

# --- Load logo and initialize backend ---
logo_path = "lexa_logo.png"
st.markdown(f'<div class="header-logo"><img src="{logo_path}" alt="Lexa Logo"></div>', unsafe_allow_html=True)
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

# --- Chat history storage ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I’m Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}]

# --- Clear chat button ---
if st.button("Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I’m Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}]

# --- Function to render each message ---
def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    cls = "assistant" if role == "assistant" else "user"
    avatar_html = f'<img src="{logo_path}" class="chat-avatar"/>' if role == "assistant" else ""
    bubble_class = f'chat-bubble {cls}'
    msg_html = f'''
    <div class="chat-message {cls}">
        {avatar_html}
        <div class="{bubble_class}">{content}</div>
    </div>
    '''
    st.markdown(msg_html, unsafe_allow_html=True)

# --- Display chat messages ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    render_message(msg)
st.markdown('</div>', unsafe_allow_html=True)

# --- Input prompt ---
if prompt := st.chat_input("Ask Lexa a legal question..."):
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message({"role": "user", "content": prompt})

    # Generate and render assistant reply
    with st.spinner("Lexa is thinking..."):
        try:
            reply = st.session_state.lexa.process_query(prompt)
        except Exception as e:
            reply = "Sorry, I encountered an error. Please try again."
            st.error(f"Error: {e}")
    st.session_state.messages.append({"role": "assistant", "content": reply})
    render_message({"role": "assistant", "content": reply})
