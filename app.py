import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import json
import os

# Load each dataset file into memory
@st.cache_data
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        content = f"{item.get('topic', '')}\n\n{item.get('definition', '')}\n\n{item.get('legal_analysis', '')}"
        if "key_cases" in item:
            for case in item["key_cases"]:
                content += f"\n\nCase: {case['case_name']}\nFacts: {case['facts']}\nDecision: {case['decision']}"
        if "statute_reference" in item:
            content += f"\n\nStatute: {item['statute_reference']}"
        documents.append(Document(page_content=content))
    return documents

# Load vector store
@st.cache_resource
def load_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# Load OpenRouter-based OpenAI model
@st.cache_resource
def load_lexa_model():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

# Smart dataset selector
def select_dataset(question):
    q = question.lower()
    if any(word in q for word in ["land", "occupancy", "lease", "certificate"]):
        return "land_law_dataset.json"
    elif any(word in q for word in ["crime", "offence", "punishment", "criminal"]):
        return "criminal_law_dataset.json"
    else:
        return "contract_law_dataset.json"

# Greeting responses
def casual_response(msg):
    greetings = ["hi", "hello", "what's up", "hey"]
    if msg.lower().strip() in greetings:
        return "👋 Hello! I'm **Lexa**, your Nigerian legal assistant. Ask me any law-related question!"
    return None

# Page config
st.set_page_config(page_title="⚖️ Lexa: Nigerian Law Analyzer", page_icon="⚖️")
st.markdown("""
    <style>
    .chat-container {
        background-color: #f7f7f7;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-user {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: right;
    }
    .chat-lexa {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ Lexa: Nigerian Law Analyzer")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask Lexa about any Nigerian legal issue...")

if user_input:
    # Greet if casual
    casual = casual_response(user_input)
    if casual:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("lexa", casual))
    else:
        dataset_path = select_dataset(user_input)
        docs = load_dataset(dataset_path)
        db = load_vectorstore(docs)
        llm = load_lexa_model()
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        with st.spinner("Lexa is thinking..."):
            result = qa_chain.run(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("lexa", result))

# Chat history display
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-user">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-lexa">{msg}</div>', unsafe_allow_html=True)
