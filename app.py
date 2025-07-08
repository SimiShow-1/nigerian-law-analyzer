import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json, os

LOGO_PATH = "lexa_logo.png"
DATASET_PATHS = ["contract_law_dataset.json", "land_law_dataset.json"]

st.set_page_config(page_title="Lexa – Nigerian Law Analyzer", layout="wide")
st.image(LOGO_PATH, width=80)
st.header("Lexa – Your Nigerian Legal AI")

@st.cache_data
def load_docs():
    docs = []
    for p in DATASET_PATHS:
        if os.path.exists(p):
            for item in json.load(open(p, encoding="utf-8")):
                title = item.get("title") or item.get("topic") or "Untitled"
                content = (
                    item.get("content")
                    or item.get("definition")
                    or item.get("legal_analysis")
                    or ""
                )
                docs.append(Document(page_content=f"{title}\n\n{content}"))
    return docs

@st.cache_resource
def load_store(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

@st.cache_resource
def load_llm():
    return ChatOpenAI(openai_api_key=st.secrets["OPENROUTER_API_KEY"], temperature=0.7)

docs = load_docs()
db = load_store(docs)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(), retriever=db.as_retriever(), return_source_documents=False
)

if "history" not in st.session_state:
    st.session_state.history = []

for r, msg in st.session_state.history:
    st.chat_message(r).write(msg)

user_input = st.chat_input("Type your question…")
if user_input:
    st.session_state.history.append(("user", user_input))
    try:
        answer = qa_chain.invoke({"query": user_input})
        reply = answer["result"]
    except Exception:
        reply = "⚠️ Oops—something went wrong. Try again or contact support."

    st.session_state.history.append(("assistant", reply))
    # re-run to display new messages
