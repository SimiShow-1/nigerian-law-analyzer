import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import json
import os

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

# Load vector database
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Load Lexa LLM using OpenRouter
@st.cache_resource
def load_lexa_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        model="openrouter/mistralai/mixtral-8x7b-instruct",
        temperature=0.5,
    )

# Custom prompt
custom_prompt = PromptTemplate.from_template("""
You are Lexa, a legal assistant trained in Nigerian law. When given a question or scenario,
you must analyze it using Nigerian legal reasoning, statutes, case law, and remedies.

Be clear, helpful, and structured like a real lawyer speaking to a layperson.

Question: {question}
=========
Context: {context}
=========
Answer:
""")

# Streamlit UI
st.set_page_config(page_title="Lexa - Nigerian Law Analyzer", page_icon="⚖️")
st.title("⚖️ Lexa - Nigerian Law Analyzer")
st.markdown("_Ask Lexa any legal question or describe a Nigerian legal issue/scenario._")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input field at bottom
with st.container():
    user_input = st.chat_input("🧑‍⚖️ Type your legal question or scenario...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Lexa is thinking..."):
            docs = load_documents()
            db = load_vectorstore(docs)
            llm = load_lexa_llm()
            retriever = db.as_retriever()

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": custom_prompt}
            )

            result = qa_chain.run(user_input)

        st.session_state.messages.append({"role": "lexa", "content": result})

# Chat layout
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("🧑‍⚖️ You").markdown(msg["content"])
    else:
        st.chat_message("🧠 Lexa").markdown(msg["content"])
