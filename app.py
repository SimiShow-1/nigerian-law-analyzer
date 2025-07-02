import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI  # ✅ updated import
from langchain_core.documents import Document
import os
import json

# 🔐 Load OpenAI API Key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Paths
DATASET_PATH = "./contract_law_dataset.json"

# Load dataset from JSON
@st.cache_data
def load_documents():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    documents = []
    for item in raw_data:
        content = f"{item.get('title', '')}\n\n{item.get('content', '')}"
        documents.append(Document(page_content=content))
    return documents

# Load embeddings and create FAISS index
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Load OpenAI Chat Model (Cloud-based)
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",  # Or "gpt-4" if you're using GPT-4
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

# Streamlit UI
st.set_page_config(page_title="⚖️ Nigerian Law Analyzer", page_icon="⚖️")
st.title("⚖️ Nigerian Law Analyzer")
st.markdown("Ask me anything about **Contract Law** in Nigeria:")

user_query = st.text_input("🧑‍⚖️ Your Question:")

if user_query:
    with st.spinner("Analyzing legal data..."):
        docs = load_documents()
        db = load_vectorstore(docs)
        llm = load_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        response = qa.invoke({"query": user_query})
        st.markdown("### 🤖 AI Response:")
        st.write(response["result"])
