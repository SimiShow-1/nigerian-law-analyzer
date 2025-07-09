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

# === MODERN CHAT UI ===
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-color: #f8fafc !important;  /* Very light blue-gray */
}

/* Chat Container */
.chat-container {
    background: transparent !important;
    padding-bottom: 100px; /* Space for input */
}

/* Your Existing Bubbles (Improved) */
.user-message {
    background: #3f51b5;  /* Professional blue */
    color: white;
    border-bottom-right-radius: 4px !important;
}

.bot-message {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2c3e50;  /* Navy accent */
}

/* Input Box (Your existing style - just ensuring visibility) */
.input-container {
    background: white;
    box-shadow: 0 -2px 15px rgba(0,0,0,0.1);
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

# === Chat Display ===
chat_placeholder = st.empty()
with chat_placeholder.container():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        message_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="message {message_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# === Input Area ===
input_placeholder = st.empty()
with input_placeholder.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="input_form", clear_on_submit=True):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            user_input = st.text_input("", key="input", label_visibility="collapsed", placeholder="Ask about Nigerian law...")
        with col2:
            submitted = st.form_submit_button("✈️", help="Send")
            st.markdown('<style>div[data-testid="stFormSubmitButton"] button {width: 100%;}</style>', unsafe_allow_html=True)
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
            full_prompt = f"You're Lexa, a Nigerian legal analyst. Provide clear, concise answers about Nigerian law. Question: {user_input}"
            reply = qa.invoke(full_prompt)
    except Exception as e:
        reply = f"Sorry, I encountered an error. Please try again."
    st.session_state.history.append(("lexa", reply))
    st.rerun()

# Auto-scroll to bottom
st.markdown("""
<script>
document.addEventListener("DOMContentLoaded", function() {
    const container = document.getElementById("chat-container");
    container.scrollTop = container.scrollHeight;
});
</script>
""", unsafe_allow_html=True)