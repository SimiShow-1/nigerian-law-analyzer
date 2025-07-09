import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json
import os

# === CONFIG ===
LOGO_PATH = "lexa_logo.png"
DATASET_PATHS = ["contract_law_dataset.json", "land_law_dataset.json"]
GREETINGS = ["hi", "hello", "hey", "what's up", "sup", "how are you"]

# === PAGE SETUP ===
st.set_page_config(page_title="Lexa – Nigerian Law Analyzer", page_icon="🧠", layout="wide")
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Your Nigerian Law Analyzer")

# === NEW: MODERN CHAT UI CSS ===
st.markdown("""
<style>
/* Main chat container */
.chat-container {
    height: 70vh;
    overflow-y: auto;
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 10px;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* Message bubbles */
.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    line-height: 1.4;
    position: relative;
    word-wrap: break-word;
}

.user-message {
    align-self: flex-end;
    background-color: #0078d4;
    color: white;
    border-bottom-right-radius: 4px;
}

.bot-message {
    align-self: flex-start;
    background-color: #f1f1f1;
    color: #333;
    border-bottom-left-radius: 4px;
}

/* Input area - sticks to bottom */
.input-container {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    max-width: 800px;
    background: white;
    padding: 10px;
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    display: flex;
    z-index: 100;
}

.input-container input {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-radius: 20px;
    outline: none;
    font-size: 16px;
}

.input-container button {
    margin-left: 10px;
    padding: 12px 20px;
    background-color: #0078d4;
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-weight: 500;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}
</style>
""", unsafe_allow_html=True)

# === Load Datasets === 
@st.cache_data
def load_documents():
    all_docs = []
    for path in DATASET_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                for i in data:
                    title = i.get("title") or i.get("topic") or "Untitled"
                    content = i.get("content") or i.get("definition") or ""
                    all_docs.append(Document(page_content=f"{title}\n\n{content}"))
    return all_docs

@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.7,
        model="mistralai/mixtral-8x7b-instruct"
    )

# === Chat History ===
if "history" not in st.session_state:
    st.session_state.history = []

# === NEW: IMPROVED CHAT DISPLAY ===
chat_placeholder = st.empty()
with chat_placeholder.container():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        message_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="message {message_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === NEW: FIXED INPUT AREA ===
input_placeholder = st.empty()
with input_placeholder.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input("", key="input", label_visibility="collapsed", placeholder="Ask Lexa about Nigerian law...")
        submitted = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

# === Response Logic ===
if submitted and user_input:
    st.session_state.history.append(("user", user_input))
    try:
        if user_input.strip().lower() in GREETINGS:
            reply = "Hello! I'm Lexa, your Nigerian law assistant. How can I help you today?"
        else:
            docs = load_documents()
            db = load_vectorstore(docs)
            qa = RetrievalQA.from_chain_type(
                llm=load_llm(),
                retriever=db.as_retriever(),
                return_source_documents=False
            )
            full_prompt = f"You're Lexa, a Nigerian legal analyst. Explain the user's query in a way that helps them understand the legal implications and remedies available. Question: {user_input}"
            reply = qa.invoke(full_prompt)
    except Exception as e:
        reply = f"Sorry, I encountered an error processing your request. Please try again."
    st.session_state.history.append(("lexa", reply))
    st.rerun()

# === NEW: AUTO-SCROLL TO BOTTOM ===
st.markdown("""
<script>
// Auto-scroll to bottom of chat
window.onload = function() {
    var container = document.getElementById("chat-container");
    container.scrollTop = container.scrollHeight;
};
</script>
""", unsafe_allow_html=True)
