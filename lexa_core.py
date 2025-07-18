import json
import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import logging

class LexaError(Exception):
    pass

class LexaCore:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = "all-MiniLM-L6-v2"
        self.llm_model = "mistralai/mixtral-8x7b-instruct"
        self.api_key = os.getenv("OPENROUTER_API_KEY") or self._get_from_streamlit()

        if not self.api_key:
            raise LexaError("No OpenRouter API key found in environment or Streamlit secrets.")

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = self._load_data()
        self.llm = ChatOpenAI(
            model=self.llm_model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.api_key
        )
        self.prompt_template = self._get_prompt_template()

    def _get_from_streamlit(self):
        try:
            import streamlit as st
            return st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            return None

    def _load_data(self) -> FAISS:
        try:
            cache_path = "faiss_index"
            if os.path.exists(cache_path):
                # ✅ Allow safe deserialization of your own FAISS index
                return FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)

            datasets = ['contract_law_dataset.json', 'land_law_dataset.json']
            documents = []

            for dataset in datasets:
                with open(dataset, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    content = item.get('content') or item.get('definition') or ""
                    title = item.get('title') or item.get('topic') or "Untitled"
                    documents.append(Document(page_content=f"{title}\n\n{content}"))

            if not documents:
                raise LexaError("No documents found to load.")

            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.save_local(cache_path)
            return vectorstore

        except Exception as e:
            raise LexaError(f"Vectorstore load failed: {e}")

    def _get_prompt_template(self) -> PromptTemplate:
        template = """
You are Lexa, a Nigerian legal assistant AI. Analyze the question below with relevant Nigerian law, organized clearly:

1. Identify the legal issue(s)
2. Explain the applicable law (cite exact Nigerian statutes when possible)
3. Apply the law to the scenario
4. Offer a brief conclusion and next steps if needed

Avoid saying you're using any method like IRAC.

Context: {context}

User Question: {query}

Lexa's Response:
"""
        return PromptTemplate(input_variables=["context", "query"], template=template)

    def process_query(self, query: str) -> str:
        try:
            greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "sup", "yo"]
            if query.strip().lower() in greetings:
                return "Hi there! I'm Lexa, your Nigerian Legal AI. Ask me anything about land law, contracts, or your rights."

            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = self.prompt_template.format(context=context, query=query)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            raise LexaError(f"Could not process the query: {e}")
