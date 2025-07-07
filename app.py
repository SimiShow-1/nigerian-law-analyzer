import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json

# Constants
DATASET_PATH = "./contract_law_dataset.json"
LOGO_PATH = "lexa_logo.png"  # Make sure this file is in the same folder
GREETINGS = ["hi", "hello", "hey", "what's up", "sup", "how are you"]

# Load legal documents
@st.cache_data
def load_documents():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=f"{i['title']}\n\n{i['content']}") for i in data]

# FAISS vector store
@st.cache_resource
def load_vectorstore(_docs):  # FIXED: underscore to avoid hash error
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

# Load the OpenRouter model
@st.cache_resource
def load_llm():
    return ChatOpenAI(openai_api_key=st.secrets["OPENREUTER_API_KEY"], temperature=0.7)

# --- UI Setup ---
st.set_page_config(page_title="Lexa - Nigerian Law Analyzer", page_icon="🧠", layout="wide")

# Display logo
try:
    st.image(LOGO_PATH, width=100)
except:
    st.warning("⚠️ Logo not found. Please add 'lexa_logo.png' to your project folder.")

st.markdown("## **Lexa**: Your Nigerian Law Analyzer")

# Styling
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #cceeff;
        color: #000000;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        font-weight: 500;
        font-size: 16px;
        align-self: flex-end;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .lexa-bubble {
        background-color: #ebebeb;
        color: #222222;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        font-weight: 500;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Chat History
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat
for role, msg in st.session_state.history:
    bubble_class = "user-bubble" if role == "user" else "lexa-bubble"
    st.markdown(f'<div class="chat-container"><div class="{bubble_class}">{msg}</div></div>', unsafe_allow_html=True)

# Chat Input
user_input = st.text_input("Type your message below...", key="user_input")

if user_input:
    st.session_state.history.append(("user", user_input))

    # Simple casual check
    if user_input.strip().lower() in GREETINGS:
        reply = "Hey! I'm Lexa. Ask me anything about Nigerian law, and I’ll break it down."
    else:
        docs = load_documents()
        db = load_vectorstore(docs)
        qa = RetrievalQA.from_chain_type(
            llm=load_llm(),
            retriever=db.as_retriever(),
            return_source_documents=False
        )
        full_prompt = f"You're Lexa, a Nigerian legal analyst. Answer concisely and legally:\n\n{user_input}"
        reply = qa.run(full_prompt)

    st.session_state.history.append(("lexa", reply))
    st.experimental_rerun()
