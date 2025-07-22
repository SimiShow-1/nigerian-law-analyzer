
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
    if 'show_clear_confirm' not in st.session_state:
        st.session_state.show_clear_confirm = False

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

# Load custom CSS
css_path = "style.css"  # Changed from styles.css to style.css
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    logger.warning("style.css not found, using default styles")
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
            background-color: #2e7d32;
            color: white;
        }
        .clear-btn {
            background-color: #d32f2f !important;
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

  

# Header
st.markdown('<div class="header" role="banner">', unsafe_allow_html=True)
try:
    st.image("lexa_logo.png", width=150)
except FileNotFoundError:
    st.image("https://via.placeholder.com/150?text=Lexa", width=150)
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
        st.session_state.show_clear_confirm = True
    
    if st.session_state.show_clear_confirm:
        st.warning("Are you sure you want to clear the conversation?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm", key="confirm_clear"):
                st.session_state.messages = []
                st.session_state.input_key += 1
                st.session_state.show_clear_confirm = False
                st.session_state.lexa.reset()  # Reset LexaCore state
                st.rerun()
        with col2:
            if st.button("Cancel", key="cancel_clear"):
                st.session_state.show_clear_confirm = False
                st.rerun()

# Suggested questions
st.markdown("### Suggested Questions")
suggested = [
    "What are the requirements for a valid contract in Nigeria?",
    "How does land ownership work under Nigerian law?",
    "What are my legal rights as a tenant?"
]
for q in suggested:
    if st.button(q, key=f"suggest_{q}"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Lexa is researching your question..."):
            response = st.session_state.lexa.process_query(q)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.input_key += 1
        st.rerun()

# Chat container
@st.fragment
def render_chat():
    with st.container():
        st.markdown('<div class="chat-container" role="log" aria-live="polite">', unsafe_allow_html=True)
        if not st.session_state.messages:
            st.markdown("No messages yet. Ask a question to start!")
        else:
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

render_chat()

# Input area
with st.form("chat_input", clear_on_submit=True):
    input_col, button_col = st.columns([10, 1])
    
    with input_col:
        user_input = st.text_input(
            "Ask a legal question...",
            key=f"input_{st.session_state.input_key}",
            placeholder="e.g. What are the requirements for a valid contract in Nigeria?",
            label_visibility="collapsed",
            help="Enter your legal question here"
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
            logger.error(f"Unexpected error: {e}")
            st.error("An unexpected error occurred. Please try again.")
