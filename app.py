import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json
import os

# Path settings
DATASET_PATH = "./contract_law_dataset.json"

# Load dataset
@st.cache_data
def load_documents():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [
        Document(page_content=f"{item.get('title', '')}\n\n{item.get('content', '')}")
        for item in raw_data
    ]

# Create vector store
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Load OpenRouter LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        model="mistralai/mistral-7b-instruct",  # or another model from OpenRouter
        temperature=0.3,
        max_tokens=512,
    )

# Streamlit UI
st.set_page_config(page_title="⚖️ Nigerian Law Analyzer", page_icon="⚖️")
st.title("⚖️ Nigerian Law Analyzer")
st.markdown("Ask me anything about **Contract Law in Nigeria**:")

user_query = st.text_input("🧑‍⚖️ Your Question:")

if user_query:
    with st.spinner("Analyzing legal documents..."):
        docs = load_documents()
        db = load_vectorstore(docs)
        llm = load_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        response = qa.invoke({"query": user_query})
        st.markdown("### 🤖 AI Response:")
        st.write(response["result"])
