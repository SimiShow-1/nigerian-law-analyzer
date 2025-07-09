import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import json
import os
from typing import List, Tuple

# === CONFIG ===
LOGO_PATH = "lexa_logo.png"
DATASET_PATHS = ["contract_law_dataset.json", "land_law_dataset.json"]
GREETINGS = ["hi", "hello", "hey", "what's up", "sup", "how are you"]
PRIMARY_COLOR = "#3f51b5"  # Professional blue

# === PAGE SETUP ===
st.set_page_config(
    page_title="Lexa – Nigerian Law Analyzer", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === MODERN UI ===
st.markdown(f"""
<style>
/* Background */
.stApp {{
    background-color: #f8fafc !important;
}}

/* Chat Container */
.chat-container {{
    height: calc(100vh - 180px);
    overflow-y: auto;
    padding: 20px 5%;
    background: transparent;
}}

/* Message Bubbles */
.message {{
    max-width: 78%;
    padding: 14px 18px;
    border-radius: 18px;
    line-height: 1.5;
    font-size: 15px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.user-message {{
    align-self: flex-end;
    background: {PRIMARY_COLOR};
    color: white;
    border-bottom-right-radius: 4px !important;
}}

.bot-message {{
    align-self: flex-start;
    background: white;
    border-left: 4px solid #2c3e50;
    color: #333;
}}

/* Input Area */
.input-container {{
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 90%;
    max-width: 800px;
    background: white;
    padding: 8px 15px;
    border-radius: 25px;
    box-shadow: 0 -2px 15px rgba(0,0,0,0.1);
    z-index: 100;
    border: 1px solid #e2e8f0;
}}

.input-container input {{
    flex: 1;
    padding: 12px 15px;
    border: none;
    font-size: 16px;
}}

/* Paper Plane Button */
.input-container button {{
    background: transparent;
    border: none;
    color: {PRIMARY_COLOR};
    font-size: 22px;
    padding: 0 0 0 12px;
    transition: transform 0.2s;
}}

.input-container button:hover {{
    transform: translateX(3px);
    color: #303f9f;
}}

/* Animations & Misc */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

::-webkit-scrollbar {{
    width: 6px;
}}

::-webkit-scrollbar-thumb {{
    background: rgba(0,0,0,0.1);
    border-radius: 3px;
}}
</style>
""", unsafe_allow_html=True)

# === LOADERS ===
@st.cache_data
def load_documents() -> List[Document]:
    """Load and preprocess legal documents"""
    all_docs = []
    for path in DATASET_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    title = item.get("title", "Untitled")
                    content = item.get("content", "")
                    all_docs.append(Document(
                        page_content=f"TITLE: {title}\nCONTENT: {content}",
                        metadata={"source": path}
                    ))
    return all_docs

@st.cache_resource
def load_vectorstore(_docs: List[Document]):
    """Create FAISS vectorstore with better chunking"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # More accurate
    )
    return FAISS.from_documents(chunks, embed)

@st.cache_resource
def load_llm():
    """Configure LLM with better parameters"""
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.5,  # More precise answers
        model="mistralai/mixtral-8x7b-instruct",
        max_tokens=2000
    )

# === CHAT LOGIC ===
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# Header
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Nigerian Law Assistant")

# Chat Display
with st.container():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        css_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="message {css_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input Area
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([0.85, 0.15])
    with cols[0]:
        user_input = st.text_input(
            "", 
            key="input",
            label_visibility="collapsed",
            placeholder="Ask about Nigerian contract or land law..."
        )
    with cols[1]:
        submitted = st.form_submit_button("✈️")

# Response Handling
if submitted and user_input.strip():
    st.session_state.history.append(("user", user_input))
    
    try:
        if user_input.strip().lower() in GREETINGS:
            reply = "Hello! I'm Lexa, your Nigerian legal assistant. How can I help you today?"
        else:
            docs = load_documents()
            db = load_vectorstore(docs)
            
            qa = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False
            )
            
            prompt = f"""You are Lexa, a Nigerian legal expert. Provide:
            1. Clear explanation of the law
            2. Relevant statutes
            3. Practical implications
            
            Question: {user_input}"""
            
            reply = qa.invoke(prompt)["result"]
            
    except Exception as e:
        reply = "⚠️ Sorry, I encountered an error. Please try again."
    
    st.session_state.history.append(("lexa", reply))
    st.rerun()

# Auto-scroll JS
st.markdown("""
<script>
window.addEventListener('load', function() {
    const chat = document.getElementById('chat-container');
    chat.scrollTop = chat.scrollHeight;
});
</script>
""", unsafe_allow_html=True)