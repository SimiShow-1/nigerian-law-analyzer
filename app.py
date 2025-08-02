import os
import streamlit as st
from lexa_core import LexaCore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="Lexa – Nigerian Legal Assistant", layout="wide")

# Custom CSS for chat bubbles and layout
st.markdown("""
    <style>
    .header-logo { display: flex; justify-content: center; margin-bottom: 20px; }
    .header-logo img { width: 150px; border-radius: 8px; }
    .chat-container { max-width: 700px; margin: auto; padding: 10px; background: #fafafa; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .chat-message { display: flex; align-items: flex-start; margin-bottom: 12px; }
    .chat-message.user { justify-content: flex-end; }
    .chat-bubble { max-width: 80%; padding: 10px 14px; border-radius: 14px; line-height: 1.4; color: #000 !important; }
    .chat-bubble.user { background: #dcf8c6; border-bottom-right-radius: 4px; text-align: right; }
    .chat-bubble.assistant { background: #ffffff; border: 1px solid #e0e0e0; border-bottom-left-radius: 4px; }
    .chat-avatar { width: 32px; height: 32px; border-radius: 50%; margin-right: 8px; }
    </style>
""", unsafe_allow_html=True)

# Header logo
st.markdown('<div class="header-logo"><img src="lexa_logo.png" alt="Lexa Logo"></div>', unsafe_allow_html=True)

# Initialize backend
if "lexa" not in st.session_state:
    try:
        st.session_state.lexa = LexaCore()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"}
    ]

# Display chat
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    content = msg["content"].replace("\n", "<br>")
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="chat-message user">
                <div class="chat-bubble user">{content}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message assistant">
                <img src="lexa_logo.png" class="chat-avatar" />
                <div class="chat-bubble assistant">{content}</div>
            </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input prompt
if prompt := st.chat_input("Ask Lexa a legal question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-bubble user">{prompt}</div>
        </div>
    """, unsafe_allow_html=True)

    # Assistant response
    with st.spinner("Lexa is thinking..."):
        try:
            reply = st.session_state.lexa.process_query(prompt)
        except Exception:
            reply = "Sorry, I encountered an error. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.markdown(f"""
        <div class="chat-message assistant">
            <img src="lexa_logo.png" class="chat-avatar" />
            <div class="chat-bubble assistant">{reply}</div>
        </div>
    """, unsafe_allow_html=True)
