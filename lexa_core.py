import json
import os
import logging
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Try importing streamlit, but make it optional
try:
    import streamlit as st
except ImportError:
    st = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
os.environ['FAISS_NO_GPU'] = '1'

class LexaError(Exception):
    pass

class LexaCore:
    def __init__(self):
        self._validate_environment()
        self.embedding_model = "all-MiniLM-L6-v2"
        self.llm_model = "mistralai/mixtral-8x7b-instruct"
        self.api_key = self._get_api_key()
        self.similarity_k = 3

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = self._load_vectorstore()
        self.llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.api_key,
            temperature=0.3,
            max_tokens=500
        )
        self.prompt_template = self._get_prompt_template()
        self.query_cache = {}

    def _validate_environment(self):
        for path in ["contract_law_dataset.json", "land_law_dataset.json"]:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                self._create_default_dataset(path)

    def _create_default_dataset(self, path: str):
        base = os.path.basename(path).split("_")[0].capitalize()
        data = [{
            "title": f"Default {base} Law",
            "content": f"This is a default {base} law document."
        }]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _get_api_key(self):
        key = None
        if st and hasattr(st, "secrets"):
            key = st.secrets.get("OPENROUTER_API_KEY")
        if not key:
            key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise LexaError("API key not found. Please set OPENROUTER_API_KEY in .streamlit/secrets.toml or as an environment variable.")
        return key

    def _load_documents(self):
        docs = []
        for path in ["contract_law_dataset.json", "land_law_dataset.json"]:
            try:
                items = json.load(open(path, "r", encoding="utf-8"))
                for it in items:
                    content = it.get("content") or it.get("definition", "")
                    if not content:
                        continue
                    title = it.get("title", it.get("term", "Untitled"))
                    docs.append(Document(page_content=f"{title}\n\n{content}", metadata={"source": path}))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        if not docs:
            raise LexaError("No documents loaded")
        return docs

    def _load_vectorstore(self):
        documents = self._load_documents()
        return FAISS.from_documents(documents, self.embeddings)

    def _get_prompt_template(self):
        template = """
You are Lexa, a Nigerian legal assistant trained on Nigerian law.
Use the provided context to define, explain, and apply relevant legal concepts.
When necessary, cite applicable Nigerian Acts, sections, and legal principles.
Don't answer questions you were'nt asked, understand the user input before responding. If the user input is not understood ask for clarification.

Context:
{context}

Question:
{question}
"""
        return PromptTemplate(input_variables=["context", "question"], template=template)

    def process_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Please ask a legal question."

        if re.match(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b", query.lower()):
            return "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"

        if query in self.query_cache:
            return self.query_cache[query]

        try:
            docs = self.vectorstore.similarity_search(query, k=self.similarity_k)
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            prompt = self.prompt_template.format(context=context, question=query)

            response = self.llm.invoke(prompt)
            text = response.content.strip()
            self.query_cache[query] = text
            return text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Sorry, I encountered an error. Please try again."

    def reset(self):
        self.query_cache.clear()
