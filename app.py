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

# === Custom CSS for WhatsApp-style Input ===
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    height: 82vh;
    overflow-y: auto;
}
.bubble {
    max-width: 75%;
    padding: 12px;
    margin-bottom: 10px;
    font-size: 16px;
    font-weight: 500;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.user-bubble {
    align-self: flex-end;
    background-color: #cceeff;
    color: #000000;
}
.lexa-bubble {
    align-self: flex-start;
    background-color: #ebebeb;
    color: #222222;
}
.input-area {
    position: fixed;
    bottom: 10px;
    left: 3%;
    right: 3%;
    background-color: #ffffff;
    padding: 5px;
    border-radius: 10px;
    display: flex;
    border: 1px solid #ccc;
}
.input-area input[type="text"] {
    flex-grow: 1;
    border: none;
    outline: none;
    padding: 10px;
    font-size: 16px;
}
.input-area button {
    border: none;
    background-color: #25D366;
    color: white;
    font-size: 18px;
    padding: 0 16px;
    border-radius: 0 10px 10px 0;
    cursor: pointer;
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
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg in st.session_state.history:
    bubble_class = "user-bubble" if role == "user" else "lexa-bubble"
    st.markdown(f'<div class="bubble {bubble_class}">{msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# === Input Bar ===
st.markdown('<div class="input-area">', unsafe_allow_html=True)
with st.form(key="input_form", clear_on_submit=True):
    col1, col2 = st.columns([0.88, 0.12])
    with col1:
        user_input = st.text_input("", key="input", label_visibility="collapsed", placeholder="Ask Lexa something...")
    with col2:
        submitted = st.form_submit_button("📤")
st.markdown('</div>', unsafe_allow_html=True)

# === Response Logic ===
if submitted and user_input:
    st.session_state.history.append(("user", user_input))
    try:
        if user_input.strip().lower() in GREETINGS:
            reply = "Hi, I'm Lexa. You can ask me anything about Nigerian Law—Contract, Land, or more."
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
        reply = f"⚠️ Lexa encountered an error:\n`{str(e)}`"
    st.session_state.history.append(("lexa", reply))
    st.rerun()

