import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import json
import os

# Paths
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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

# Load vectorstore
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.from_documents(chunks, embeddings)

# Load cloud-based LLM (OpenRouter using ChatOpenAI wrapper)
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model_name="mistralai/mixtral-8x7b-instruct",
        openai_api_key="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",  # Replace with your actual OpenRouter key
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=512,
    )

# UI
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
