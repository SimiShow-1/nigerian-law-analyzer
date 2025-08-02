import streamlit as st
from streamlit_chat import message
import time

st.set_page_config(page_title="Lexa", page_icon="🤖")
st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.8rem;
    }
    .stChatMessage img {
        height: 60px !important; /* bigger avatar */
        width: 60px !important;
        border-radius: 50%;
    }
    </style>
""", unsafe_allow_html=True)

# Set up session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and logo
col1, col2 = st.columns([1, 8])
with col1:
    st.image("lexa_logo.png", width=60)
with col2:
    st.title("Lexa")

# Display chat history
for idx, msg in enumerate(st.session_state.chat_history):
    is_user = msg["role"] == "user"
    message(
        msg["content"],
        is_user=is_user,
        key=f"chat_{idx}",
        avatar_style="thumbs" if is_user else None,
        avatar_url=None if is_user else "lexa_logo.png"
    )

# Input field
prompt = st.chat_input("Say something to Lexa...")

if prompt:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Rerun to immediately display user message
    st.experimental_rerun()

# Only respond after rerun
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "user":
        # Simulate Lexa's reply
        with st.spinner("Lexa is thinking..."):
            time.sleep(1)  # fake delay

        reply = f"Lexa heard: '{last_msg['content']}'"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Rerun to display assistant message
        st.rerun()

