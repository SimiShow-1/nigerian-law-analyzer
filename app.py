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

# === PERFECT CHAT UI ===
st.markdown(f"""
<style>
/* === FIXED STRUCTURE === */
html, body, #root, .stApp {{
    height: 100%;
    overflow: hidden;
}}

/* Main chat area - scrollable */
[data-testid="stAppViewContainer"] > .main {{
    padding-bottom: 80px !important;
}}

/* Chat bubbles */
.message {{
    max-width: 80%;
    padding: 14px 18px;
    border-radius: 18px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease;
    line-height: 1.5;
}}

.user-message {{
    background: {PRIMARY_COLOR};
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}}

.bot-message {{
    background: #f1f1f1;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}}

/* === YOUR REQUESTED INPUT STYLE === */
.input-container {{
    position: fixed !important;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 12px 5%;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    z-index: 999;
    border-top: 1px solid #e0e0e0;
    display: flex;
}}

.input-box {{
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-right: none;
    border-radius: 24px 0 0 24px;
    font-size: 16px;
    outline: none;
}}

.send-button {{
    background: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 0 24px 24px 0;
    padding: 0 20px;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
}}

.send-button:hover {{
    background: #303f9f;
}}

/* Background */
.stApp {{
    background-color: #f5f7fb;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)

# === OPTIMIZED LOADERS ===
@st.cache_data(show_spinner=False)
def load_documents() -> List[Document]:
    all_docs = []
    for path in DATASET_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                all_docs.extend(
                    Document(
                        page_content=f"{item.get('title', 'Untitled')}\n\n{item.get('content', '')}",
                        metadata={"source": path}
                    )
                    for item in data
                )
    return all_docs

@st.cache_resource(show_spinner=False)
def load_vectorstore(_docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.6,
        model="anthropic/claude-3-haiku",
        max_tokens=1500,
        streaming=False
    )

# === CHAT IMPLEMENTATION ===
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = [] # type: ignore

# Header
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Nigerian Law Assistant")

# Chat Display
chat_placeholder = st.empty()
with chat_placeholder.container():
    for role, msg in st.session_state.history:
        css_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="message {css_class}">{msg}</div>', unsafe_allow_html=True)

# Fixed Input Area (Your Requested Style)
input_placeholder = st.empty()
with input_placeholder.container():
    st.markdown("""
    <div class="input-container">
        <input class="input-box" id="user_input" placeholder="Ask about Nigerian law..." type="text">
        <button class="send-button" onclick="document.getElementById('chat_form').submit()">➤</button>
    </div>
    <form id="chat_form" style="display: none;"></form>
    """, unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("", key="input", label_visibility="collapsed")
        submitted = st.form_submit_button("Submit")

# Response Handling
if submitted and user_input.strip():
    st.session_state.history.append(("user", user_input))
    
    with st.spinner("Lexa is researching..."):
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
                    verbose=False
                )
                
                reply = qa.run(
                    f"Answer as a Nigerian legal expert. Be concise but thorough.\n"
                    f"Question: {user_input}"
                )
                
        except Exception as e:
            reply = "Apologies, I'm temporarily unavailable. Please try again shortly."
        
        st.session_state.history.append(("lexa", reply))
        st.rerun()

# Auto-scroll JS
st.markdown("""
<script>
function scrollToBottom() {
    const container = window.parent.document.querySelector(".stChatMessage");
    if (container) container.scrollTop = container.scrollHeight;
}
window.addEventListener("load", scrollToBottom);
window.addEventListener("message", (event) => {
    if (event.data.type === "streamlit:componentEvent") scrollToBottom();
});
</script>
""", unsafe_allow_html=True)