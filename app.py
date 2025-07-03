import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import json
import os

# Load your contract law dataset
@st.cache_data
def load_documents():
    with open("contract_law_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    documents = []
    for item in raw_data:
        content = f"{item.get('title', '')}\n\n{item.get('content', '')}"
        documents.append(Document(page_content=content))
    return documents

# Vectorstore creation
@st.cache_resource
def load_vectorstore(_docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# LLM setup (OpenRouter)
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model_name="mistralai/mixtral-8x7b",  # You can switch to any OpenRouter-supported model
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    )

# Prompt template (for improved legal reasoning)
LEGAL_PROMPT_TEMPLATE = """
You are Lexa, an expert Nigerian legal analyst. Break down the user's scenario clearly using Nigerian laws.
Use specific sections of the law, relevant legal principles, and cite Nigerian cases and statutes when appropriate.

Always follow this format:

1. **Legal Issue**
2. **Applicable Law**
3. **Application to Facts**
4. **Conclusion / Advice**

User's question: {query}
"""

# Streamlit App UI
st.set_page_config(page_title="Lexa - Nigerian Law Analyzer", page_icon="⚖️")
st.markdown("<h1 style='text-align: center;'>⚖️ Lexa: Nigerian Law Analyzer</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load tools
docs = load_documents()
db = load_vectorstore(docs)
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Session history for messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input at bottom
user_input = st.chat_input("Ask Lexa a Nigerian legal question...")

# Display messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Lexa is analyzing the legal question..."):
            full_prompt = LEGAL_PROMPT_TEMPLATE.format(query=user_input)
            result = qa_chain.run(full_prompt)
            st.markdown(result)
            st.session_state.chat_history.append({"role": "assistant", "content": result})
