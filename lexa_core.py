
import json
import os
import logging
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

os.environ['FAISS_NO_GPU'] = '1'

class LexaError(Exception):
    pass

class LexaCore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._validate_environment()
        self.embedding_model = "all-MiniLM-L6-v2"
        self.llm_model = "mistralai/mixtral-8x7b-instruct"
        self.api_key = self._get_api_key()
        self.similarity_k = 3
        
        self.logger.info("Initializing embeddings")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.logger.info("Creating vectorstore")
        self.vectorstore = self._load_vectorstore()
        self.logger.info("Initializing LLM")
        self.llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.api_key,
            temperature=0.3,
            max_tokens=500
        )
        self.logger.info("Creating prompt template")
        self.prompt_template = self._get_prompt_template()
        self.query_cache = {}
        self.logger.info("LexaCore initialized")

    def _validate_environment(self):
        required_files = ["contract_law_dataset.json", "land_law_dataset.json"]
        for file in required_files:
            if not os.path.exists(file):
                self.logger.info(f"Creating default dataset: {file}")
                self._create_default_dataset(file)
            if os.path.getsize(file) == 0:
                raise LexaError(f"Data file is empty: {file}")

    def _create_default_dataset(self, file_path: str):
        default_data = [
            {
                "title": f"Default {os.path.basename(file_path).split('_')[0].capitalize()} Law",
                "content": f"This is a default {os.path.basename(file_path).split('_')[0]} law document for testing."
            }
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2)

    def _get_api_key(self):
        api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            self.logger.error("API key not found in st.secrets or environment variable")
            raise LexaError("API key not found")
        self.logger.info("API key loaded successfully")
        return api_key

    def _load_vectorstore(self):
        cache_path = "faiss_index"
        documents = self._load_documents()
        if not documents:
            raise LexaError("No legal documents loaded")
        self.logger.info(f"Creating FAISS index with {len(documents)} documents")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(cache_path)
        return vectorstore

    def _load_documents(self):
        datasets = ['contract_law_dataset.json', 'land_law_dataset.json']
        documents = []
        for dataset in datasets:
            try:
                self.logger.info(f"Loading dataset: {dataset}")
                with open(dataset, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    self.logger.warning(f"Invalid dataset format: {dataset}")
                    continue
                for item in data:
                    content = item.get("content", "") or item.get("definition", "")
                    if not content:
                        continue
                    title = item.get("title", item.get("term", "Untitled"))
                    topic = item.get("topic", item.get("term", os.path.basename(dataset).split('_')[0]))
                    documents.append(Document(
                        page_content=f"{title}\n\n{content}",
                        metadata={'source': dataset, 'title': title, 'topic': topic}
                    ))
                self.logger.info(f"Loaded {len(documents)} documents from {dataset}")
            except Exception as e:
                self.logger.warning(f"Failed to load {dataset}: {e}")
                continue
        if not documents:
            raise LexaError("No valid legal documents loaded")
        return documents

    def _get_prompt_template(self):
        template = """You are Lexa, a Nigerian legal assistant. Answer the question based on the provided context."""
        return PromptTemplate(input_variables=["context"], template=template)

    def process_query(self, query: str):
        query = query.strip()
        if not query:
            return "Please ask a legal question."
        if self._is_greeting(query):
            return "Hello! I'm Lexa, your Nigerian legal assistant. Ask me about Contract Law or Land Law!"
        if query in self.query_cache:
            return self.query_cache[query]
        self.logger.info(f"Processing query: {query[:50]}...")
        docs = self.vectorstore.similarity_search(query, k=self.similarity_k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt = self.prompt_template.format(context=context)
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()
        self.query_cache[query] = response_text
        return response_text

    def _is_greeting(self, text: str):
        return bool(re.match(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b", text.lower()))

    def reset(self):
        self.query_cache.clear()
        self.logger.info("LexaCore reset")
