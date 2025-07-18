import streamlit as st
from streamlit_chat import message
from lexa_core import LexaCore, LexaError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

try:
    lexa = LexaCore()
except LexaError as e:
    st.error(f"❌ Failed to initialize Lexa: {str(e)}")
    st.stop()

st.set_page_config(page_title="Lexa - Nigerian Legal AI Assistant", page_icon="⚖️", layout="wide")

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
        padding-right: 40px;
    }
    .send-button {
        position: absolute;
        right: 15px;
        top: 50%;
        transform: translateY(-50%);
        background: none;
        border: none;
        cursor: pointer;
    }
    .user-message {
        background-color: #DCF8C6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .lexa-message {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.image("lexa_logo.png", width=150)
st.title("Lexa - Your Nigerian Legal AI Assistant")

chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            message(msg['content'], is_user=True, key=f"user_{i}", avatar_style="initials", seed="You")
        else:
            message(msg['content'], is_user=False, key=f"lexa_{i}", avatar_style="bottts")
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([10, 1])
    with col1:
        user_input = st.text_input("Ask a legal question...", key=f"input_{st.session_state.input_key}")
    with col2:
        st.markdown('<button type="submit" class="send-button">✈️</button>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if user_input:
        try:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Lexa is thinking..."):
                response = lexa.process_query(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.input_key += 1
            st.rerun()

        except LexaError as e:
            st.error(f"Lexa Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error("An unexpected error occurred. Please try again.")
