import streamlit as st
from openai import OpenAI
import os

# Load API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set Streamlit page config
st.set_page_config(page_title="Lexa", page_icon="🤖")

# Custom CSS for chat bubbles and avatars
st.markdown("""
    <style>
        .chat-message {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        .chat-message.user {
            justify-content: flex-end;
        }
        .chat-message.assistant {
            justify-content: flex-start;
        }
        .chat-bubble {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .chat-bubble.user {
            background-color: #DCF8C6;
            color: #000;
            border-bottom-right-radius: 0;
        }
        .chat-bubble.assistant {
            background-color: #F1F0F0;
            color: #000;
            border-bottom-left-radius: 0;
        }
        .chat-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .chat-container {
            padding: 10px 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and branding
st.markdown("<h2 style='text-align: center;'>Lexa 🤖</h2>", unsafe_allow_html=True)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I’m Lexa, how can I help you today?"}
    ]

# Function to render messages
def render_message(message):
    role = message["role"]
    content = message["content"]
    if role == "user":
        html = f"""
        <div class="chat-container">
            <div class="chat-message user">
                <div class="chat-bubble user">{content}</div>
            </div>
        </div>
        """
    else:
        html = f"""
        <div class="chat-container">
            <div class="chat-message assistant">
                <img src="lexa_logo.png" class="chat-avatar" />
                <div class="chat-bubble assistant">{content}</div>
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

# Display all chat messages
for msg in st.session_state.messages:
    render_message(msg)

# Input box
if prompt := st.chat_input("Ask Lexa anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Lexa is thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            render_message({"role": "user", "content": prompt})
            render_message({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Something went wrong: {e}")
