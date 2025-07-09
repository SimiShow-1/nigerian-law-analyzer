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
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
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
        chunk_size=1000,  # Smaller chunks for faster retrieval
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\. )"]
    )
    chunks = splitter.split_documents(_docs)
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Faster than GPU for small models
    )
    return FAISS.from_documents(chunks, embed)

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.3,  # More deterministic
        model="anthropic/claude-3-haiku",  # Fastest quality model
        max_tokens=1000,  # Limit for quicker responses
        request_timeout=30  # Fail fast
    )

# === STREAMLIT UI ===
# (Keep your existing CSS and layout code from previous versions)

# === CHAT LOGIC ===
if "history" not in st.session_state:
    st.session_state.history = []

# ... (previous UI setup code)

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
                    retriever=db.as_retriever(
                        search_kwargs={"k": 2}  # Fewer docs = faster
                    ),
                    verbose=False
                )
                
                # Use the optimized prompt
                reply = qa.run(build_prompt(user_input))
                
                # Post-process for clarity
                reply = reply.replace("**IRAC Analysis**", "## Legal Analysis")
                
        except Exception as e:
            reply = f"Apologies, I'm currently overloaded. Please rephrase or try again later."
        
        st.session_state.history.append(("lexa", reply))
        st.rerun()