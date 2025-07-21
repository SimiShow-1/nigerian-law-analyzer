import streamlit as st
from streamlit_chat import message
from lexa_core import LexaCore, LexaError
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

init_session_state()

# Initialize LexaCore
@st.cache_resource(show_spinner=False)
def initialize_lexa():
    try:
        return LexaCore()
    except LexaError as e:
        st.error(f"❌ Failed to initialize Lexa: {str(e)}")
        st.stop()

if 'lexa' not in st.session_state:
    with st.spinner("Initializing Lexa legal assistant..."):
        st.session_state.lexa = initialize_lexa()

# Page configuration
st.set_page_config(
    page_title="Lexa - Nigerian Legal AI Assistant", 
    page_icon="⚖️", 
    layout="wide"
)

# Custom CSS
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
        z-index: 100;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 12px 20px;
    }
    .stButton button {
        border-radius: 20px;
        padding: 8px 20px;
        background-color: #4CAF50;
        color: white;
    }
    .clear-btn {
        background-color: #f44336 !important;
    }
    .header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.image("lexa_logo.png", width=150)
st.title("Lexa - Your Nigerian Legal AI Assistant")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About Lexa")
    st.markdown("""
    Lexa is an AI legal assistant specializing in Nigerian law.
    Ask questions about:
    - Contract Law
    - Land Law
    - Legal Rights
    - Court Procedures
    """)
    
    if st.button("Clear Conversation", key="clear", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.input_key += 1
        st.rerun()

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            message(
                msg['content'], 
                is_user=True, 
                key=f"user_{i}", 
                avatar_style="initials", 
                seed="You"
            )
        else:
            message(
                msg['content'], 
                is_user=False, 
                key=f"lexa_{i}", 
                avatar_style="bottts",
                seed="Lexa"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input area
with st.form("chat_input", clear_on_submit=True):
    input_col, button_col = st.columns([10, 1])
    
    with input_col:
        user_input = st.text_input(
            "Ask a legal question...",
            key=f"input_{st.session_state.input_key}",
            placeholder="e.g. What are the requirements for a valid contract in Nigeria?",
            label_visibility="collapsed"
        )
    
    with button_col:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input:
        try:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Lexa is researching your question..."):
                response = st.session_state.lexa.process_query(user_input)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.input_key += 1
            st.rerun()

        except LexaError as e:
            st.error(f"Lexa Error: {str(e)}")
            logger.error(f"Lexa Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error("An unexpected error occurred. Please try again.")