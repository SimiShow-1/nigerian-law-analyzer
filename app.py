import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json, os

# === CONFIG ===
LOGO_PATH = "lexa_logo.png"
DATASET_PATHS = ["contract_law_dataset.json", "land_law_dataset.json"]
GREETINGS = ["hi", "hello", "hey", "what's up", "sup", "how are you"]

# === STREAMLIT SETUP ===
st.set_page_config("Lexa - Nigerian Law Analyzer", page_icon="🧠", layout="wide")
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Your Nigerian Law Analyzer")

# === STYLES ===
st.markdown("""
<style>
.user-bubble {
  background-color: #cceeff;
  color: #000;
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
  color: #222;
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
    font-size: 22px;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    margin-top: 0.2rem;
}
button[kind="secondaryFormSubmit"]:hover {
    background-color: #1DA851;
}
button[kind="secondaryFormSubmit"]::before {
    content: "📨";
    font-size: 1.3rem;
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
                for item in json.load(f):
                    title = item.get("title") or item.get("topic") or "Untitled"
                    content = item.get("content") or item.get("definition") or ""
                    all_docs.append(Document(page_content=f"{title}\n\n{content}"))
    return all_docs

# === Vectorstore Caching ===
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# === Load OpenRouter LLM ===
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.7,
        model="mistralai/mixtral-8x7b-instruct"
    )

# === Init chat history ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Load AI Tools once ===
docs = load_documents()
db = load_vectorstore(docs)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    retriever=db.as_retriever(),
    return_source_documents=False
)

# === Show chat bubbles ===
for role, msg in st.session_state.history:
    bubble = "user-bubble" if role == "user" else "lexa-bubble"
    st.markdown(f'<div class="{bubble}">{msg}</div>', unsafe_allow_html=True)

# === Chat input form ===
with st.form(key="chat_form", clear_on_submit=True):
    st.markdown("""
    <style>
    .custom-input input {
        border: 1px solid #ccc;
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
        width: 100%;
    }
    .custom-send-btn {
        background-color: #25D366;
        border: none;
        color: white;
        padding: 0.55rem 0.8rem;
        font-size: 18px;
        border-radius: 50%;
        cursor: pointer;
        transition: background-color 0.2s;
        margin-left: 0.5rem;
    }
    .custom-send-btn:hover {
        background-color: #1DA851;
    }
    </style>
    """, unsafe_allow_html=True)

    input_col, btn_col = st.columns([0.88, 0.12])
    with input_col:
        user_input = st.text_input("Ask Lexa about Nigerian law (e.g., land disputes, breach of contract)", key="input", label_visibility="collapsed", placeholder="Ask Lexa about Nigerian law...")
    with btn_col:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and user_input:
    st.session_state.history.append(("user", user_input))
    try:
        if user_input.lower().strip() in GREETINGS:
            reply = "Hi, I'm Lexa. Ask me anything about Nigerian Law — Land, Contract, or others!"
        else:
            prompt = (
                f"You're Lexa, a Nigerian legal AI. Give a clear legal breakdown of this question, "
                f"explain any laws or remedies involved, and keep it accurate:\n\n{user_input}"
            )
            reply = qa_chain.run(prompt)
    except Exception as e:
        reply = f"⚠️ Lexa encountered an error:\n`{str(e)}`"

    st.session_state.history.append(("lexa", reply))
    st.rerun()
