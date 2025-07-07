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

# === Streamlit Page Config ===
st.set_page_config(page_title="Lexa - Nigerian Law Analyzer", page_icon="🧠", layout="wide")
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Your Nigerian Law Analyzer")

# === Custom CSS ===
st.markdown("""
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
button[kind="secondaryFormSubmit"] {
    background-color: #25D366;
    border: none;
    color: white;
    font-size: 20px;
    padding: 0.4rem 0.8rem;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
button[kind="secondaryFormSubmit"]:hover {
    background-color: #1DA851;
}
button[kind="secondaryFormSubmit"]::before {
    content: "📤";
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# === Load & Merge Datasets ===
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

# === Embeddings & Vectorstore ===
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

# === Load OpenRouter LLM ===


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.7,
        model="mistralai/mixtral-8x7b-instruct"  # You can use another supported OpenRouter model
    )

# === Chat History ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Display Chat Bubbles ===
for role, msg in st.session_state.history:
    bubble = "user-bubble" if role == "user" else "lexa-bubble"
    st.markdown(f'<div class="{bubble}">{msg}</div>', unsafe_allow_html=True)

# === Chat Input & Send Button ===
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        user_input = st.text_input("Type your question...", key="input", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button(" ", use_container_width=True)

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
            reply = qa.run(full_prompt)
    except Exception as e:
        reply = f"⚠️ Lexa encountered an error:\n`{str(e)}`"
    
    st.session_state.history.append(("lexa", reply))
    st.rerun()
