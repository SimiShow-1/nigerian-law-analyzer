import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import json
import os

# Load your legal dataset
@st.cache_data
def load_documents():
    with open("contract_law_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    documents = []
    for item in raw_data:
        content = f"{item.get('title', '')}\n\n{item.get('content', '')}"
        documents.append(Document(page_content=content))
    return documents

# Setup embeddings and vector store
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Setup OpenRouter LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct", 
        temperature=0.3
    )

# Streamlit UI setup
st.set_page_config(page_title="⚖️ Lexa: Nigerian Law Analyzer", page_icon="⚖️")
st.title("⚖️ Lexa: Nigerian Law Analyzer")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat-style interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Lexa about Nigerian law (e.g., contract disputes, land use, etc.)")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Lexa is thinking..."):

            # Handle greetings
            if user_input.strip().lower() in ["hi", "hello", "hey", "good morning", "good afternoon"]:
                response = "Hello! I'm **Lexa**, your Nigerian Law assistant. Ask me any legal question — for example, *'What is a valid contract?'*"
            else:
                # Load everything
                docs = load_documents()
                db = load_vectorstore(docs)
                llm = load_llm()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever(),
                    chain_type="stuff"
                )

                # Legal prompting logic
                prompt = f"""You are Lexa, an expert Nigerian legal assistant. 
You explain legal questions clearly using Nigerian law. Include sections of the law, case references, and useful advice if applicable.

Question: {user_input}
"""
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]

        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
