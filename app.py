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
PRIMARY_COLOR = "#3f51b5"

# === PROMPT ENGINEERING ===
def build_prompt(question: str) -> str:
    return f"""
As Lexa, Nigeria's premier AI legal assistant, follow these rules:

1. **Question Analysis**:
   - Is scenario? {"Yes" if any(x in question.lower() for x in ["if", "scenario", "case", "what would"]) else "No"}
   - Is definition? {"Yes" if any(x in question.lower() for x in ["what is", "define", "explain"]) else "No"}

2. **Response Format**:
   {"**IRAC Format** (Issue, Rule, Application, Conclusion)" if any(x in question.lower() for x in ["if", "scenario", "case"]) else 
    "**Definition Format** (Core meaning, Statutes, Example)"}

3. **Nigerian Focus**:
   - Cite exact laws (e.g. "Section 4 of Lagos Tenancy Law 2011")
   - Prioritize federal > state > local laws
   - Include practical next steps

**Question**: {question}

**Lexa's Response**:
"""

# === OPTIMIZED LOADERS ===
@st.cache_data(show_spinner=False, ttl=3600)
def load_documents() -> List[Document]:
    docs = []
    for path in DATASET_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                docs.extend(
                    Document(
                        page_content=f"{item.get('title', '')}\n\n{item.get('content', '')}",
                        metadata={"source": path, "type": "legal_text"}
                    )
                    for item in data
                )
    return docs

@st.cache_resource(show_spinner=False)
def load_vectorstore(_docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\. )"]
    )
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return FAISS.from_documents(chunks, embed)

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.3,
        model="anthropic/claude-3-haiku",
        max_tokens=1000,
        request_timeout=30
    )

# === STREAMLIT UI ===
st.set_page_config(
    page_title="Lexa – Nigerian Law Analyzer", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(f"""
<style>
.stApp {{
    background-color: #172d57 !important;
}}
.chat-container {{
    height: calc(100vh - 180px);
    overflow-y: auto;
    padding: 20px 5%;
}}
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
    background: #291616;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}}
.input-container {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 12px 5%;
    z-index: 999;
    border-top: 1px solid #2a3a5a;
}}
div.stTextInput > div > div > input {{
    padding: 12px 16px !important;
    border-radius: 24px 0 0 24px !important;
    border-right: none !important;
}}
div.stTextInput > div > div {{
    display: flex !important;
}}
div.stTextInput button {{
    border-radius: 0 24px 24px 0 !important;
    background: {PRIMARY_COLOR} !important;
    color: white !important;
    border: none !important;
    margin-left: 0 !important;
    padding: 0 20px !important;
}}
@media (max-width: 600px) {{
    .input-container {{ padding: 8px 3% !important; }}
    div.stTextInput > div > div > input {{ padding: 10px 12px !important; }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)

# === CHAT LOGIC ===
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.image(LOGO_PATH, width=100)
st.markdown("## **Lexa**: Nigerian Law Assistant")

# Chat Display
chat_placeholder = st.empty()
with chat_placeholder.container():
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        css_class = "user-message" if role == "user" else "bot-message"
        st.markdown(f'<div class="message {css_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input Area
input_placeholder = st.empty()
with input_placeholder.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            user_input = st.text_input(
                "", 
                key="input", 
                label_visibility="collapsed", 
                placeholder="Ask about Nigerian law..."
            )
        with cols[1]:
            submitted = st.form_submit_button("➤")
    st.markdown('</div>', unsafe_allow_html=True)

# Response Handling
if submitted and user_input.strip():
    st.session_state.history.append(("user", user_input))
    
    with st.spinner("Lexa is analyzing..."):
        try:
            if user_input.strip().lower() in GREETINGS:
                reply = "Hello! I'm Lexa, your Nigerian legal assistant. How can I help you today?"
            else:
                docs = load_documents()
                db = load_vectorstore(docs)
                
                qa = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 2}),
                    verbose=False
                )
                
                reply = qa.run(build_prompt(user_input))
                reply = reply.replace("**IRAC Analysis**", "## Legal Analysis")
                
        except Exception as e:
            reply = "Apologies, I'm currently overloaded. Please try again shortly."
        
        st.session_state.history.append(("lexa", reply))
        st.rerun()

# Auto-scroll JS
st.markdown("""
<script>
function scrollToBottom() {
    const container = document.getElementById("chat-container");
    if (container) container.scrollTop = container.scrollHeight;
}
new MutationObserver(scrollToBottom).observe(
    document.getElementById("chat-container"), 
    { childList: true }
);
</script>
""", unsafe_allow_html=True)