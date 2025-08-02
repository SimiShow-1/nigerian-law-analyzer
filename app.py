import streamlit as st
from streamlit_chat import message
from datetime import datetime

st.set_page_config(page_title="Lexa", layout="centered")

# Bigger logo at the top
st.image("lexa_logo.png", width=120)
st.markdown("<h2 style='text-align: center;'>Lexa — Nigerian Law Chatbot</h2>", unsafe_allow_html=True)

# Session state to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
prompt = st.chat_input("Ask Lexa about Nigerian law...")

if prompt:
    # Save user's message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulate Lexa's response (replace this with your model/inference logic)
    response = f"Lexa's reply to: '{prompt}'"  # Replace with real logic
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display messages
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    if is_user:
        message(msg["content"], is_user=True)
    else:
        # Show Lexa's logo before the message
        st.image("lexa_logo.png", width=40)
        message(msg["content"], is_user=False)
