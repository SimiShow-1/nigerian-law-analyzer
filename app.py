import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json
import os

# --- Constants ---
CONTRACT_PATH = "./contract_law_dataset.json"
LAND_PATH = "./land_law_dataset.json"
LOGO_PATH = "lexa_logo.png"
GREETINGS = ["hi", "hello", "hey", "what's up", "sup", "how are you"]

# --- Function to decide topic ---
def detect_topic(user_input):
    keywords_contract = ["contract", "agreement", "breach", "consideration", "offer", "acceptance"]
    keywords_land = ["land", "occupancy", "usufruct", "c of o", "right of occupancy", "land use"]

    user_input_lower = user_input.lower()
    if any(word in user_input_lower for word in keywords_land):
        return "land"
    return "contract"

# --- Load Docs ---
@st.cache_data
def load_documents(dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=f"{i['title']}\n\n{i['content']}") for i in data]

# --- Vector Store ---
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

# --- Load LLM ---
@st.cache_resource
def load_llm():
    return ChatOpenAI(openai_api_key=st.secrets["OPENREUTER_API_KEY"], temperature=0.7)

# --- UI ---
st.set_page_config(page_title="Lexa - Nigerian Law Analyzer", page_icon="🧠", layout="wide")
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Your Nigerian Law Analyzer")

# --- Styling ---
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
</style>
""", unsafe_allow_html=True)

# --- Chat History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display chat history ---
for role, msg in st.session_state.history:
    bubble = "user-bubble" if role == "user" else "lexa-bubble"
    st.markdown(f'<div class="{bubble}">{msg}</div>', unsafe_allow_html=True)

# --- Input field ---
user_input = st.text_input("Type your question...")

if user_input:
    st.session_state.history.append(("user", user_input))
    lower = user_input.strip().lower()

    if lower in GREETINGS:
        reply = "Hi, I'm Lexa. You can ask me anything about Nigerian Law—Contract, Land, or more."
    else:
        try:
            topic = detect_topic(user_input)
            dataset = CONTRACT_PATH if topic == "contract" else LAND_PATH
            docs = load_documents(dataset)
            db = load_vectorstore(docs)
            llm = load_llm()

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(),
                return_source_documents=False
            )

            full_prompt = f"You are Lexa, a professional Nigerian law assistant. Answer this query clearly and legally:\n\n{user_input}"
            reply = qa.run(full_prompt)

        except Exception as e:
            reply = f"⚠️ Lexa encountered an error:\n\n`{str(e)}`"

    st.session_state.history.append(("lexa", reply))
