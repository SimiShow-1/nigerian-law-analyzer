import json
import os
import logging
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import streamlit as st

# Disable GPU Faiss warnings
os.environ['FAISS_NO_GPU'] = '1'

class LexaError(Exception):
    """Custom exception for Lexa-specific errors"""
    pass

class LexaCore:
    def __init__(self):
        """Initialize the Lexa legal AI assistant"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._validate_environment()
        
        # Model configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model = os.getenv("LLM_MODEL", "mistralai/mixtral-8x7b-instruct")
        self.api_key = self._get_api_key()
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = self._load_vectorstore()
        self.llm = self._initialize_llm()
        self.prompt_template = self._get_prompt_template()
        self.logger.info("LexaCore initialized successfully")

    def _validate_environment(self) -> None:
        """Validate required environment setup"""
        required_files = ["contract_law_dataset.json", "land_law_dataset.json"]
        for file in required_files:
            if not os.path.exists(file):
                raise LexaError(f"Required data file not found: {file}")

    def _get_api_key(self) -> str:
        """Retrieve API key from Streamlit secrets or environment"""
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise LexaError("API key not found in configuration")
            return api_key
        except Exception as e:
            self.logger.error(f"API key retrieval failed: {e}")
            raise LexaError("API key configuration missing")

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM client"""
        return ChatOpenAI(
            model=self.llm_model,
            openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
            openai_api_key=self.api_key,
            temperature=0.7,
            max_tokens=2000
        )

    def _load_vectorstore(self) -> FAISS:
        """Load or create the FAISS vectorstore"""
        cache_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        
        try:
            if os.path.exists(cache_path):
                self.logger.info("Loading existing FAISS index")
                return FAISS.load_local(
                    cache_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )

            self.logger.info("Creating new FAISS index")
            documents = self._load_documents()
            
            if not documents:
                raise LexaError("No legal documents loaded")

            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.save_local(cache_path)
            return vectorstore

        except Exception as e:
            self.logger.error(f"Vectorstore initialization failed: {e}")
            raise LexaError("Failed to initialize legal knowledge base")

    def _load_documents(self) -> List[Document]:
        """Load and process legal documents"""
        datasets = ['contract_law_dataset.json', 'land_law_dataset.json']
        documents = []
        
        for dataset in datasets:
            try:
                with open(dataset, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    content = self._extract_content(item)
                    title = self._extract_title(item)
                    metadata = self._extract_metadata(item, dataset)
                    
                    documents.append(Document(
                        page_content=f"{title}\n\n{content}",
                        metadata=metadata
                    ))
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {dataset}: {e}")
                continue
                
        return documents

    def _extract_content(self, item: Dict[str, Any]) -> str:
        """Extract content from document item"""
        return (item.get('content') or item.get('definition') 
                or item.get('text') or item.get('summary') or "")

    def _extract_title(self, item: Dict[str, Any]) -> str:
        """Extract title from document item"""
        return (item.get('title') or item.get('topic') 
                or item.get('name') or "Untitled")

    def _extract_metadata(self, item: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Extract and clean metadata"""
        metadata = {
            'source': source,
            'title': self._extract_title(item),
            **{k: v for k, v in item.items() 
              if k not in ['content', 'definition', 'text', 'summary', 'title', 'topic', 'name']}
        }
        return {k: v for k, v in metadata.items() if v is not None}

    def _get_prompt_template(self) -> PromptTemplate:
        """Return the structured legal analysis prompt template"""
        template = """You are Lexa, a Nigerian legal assistant. Provide thorough legal analysis:

1. **Legal Issue**: Identify the core legal question
2. **Applicable Law**: Cite relevant Nigerian statutes/cases
3. **Analysis**: Apply law to the facts
4. **Conclusion**: Practical guidance

Context: {context}

Question: {query}

Respond in clear, professional language suitable for legal professionals."""
        return PromptTemplate(input_variables=["context", "query"], template=template)

    def process_query(self, query: str) -> str:
        """Process a legal query and return response"""
        try:
            query = query.strip()
            if not query:
                return "Please ask a legal question about Nigerian law."
                
            if self._is_greeting(query):
                return self._get_greeting_response()

            self.logger.info(f"Processing query: {query[:50]}...")
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            prompt = self.prompt_template.format(
                context=context,
                query=query
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise LexaError("Sorry, I encountered an error. Please rephrase your question.")

    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting"""
        greetings = ["hi", "hello", "hey", "good morning", 
                   "good afternoon", "good evening", "sup", "yo"]
        return text.lower().split()[0] in greetings

    def _get_greeting_response(self) -> str:
        """Return appropriate greeting response"""
        return ("Hello! I'm Lexa, your Nigerian legal assistant. "
               "I can help with:\n\n"
               "- Contract Law\n- Land Law\n- Legal Rights\n- Court Procedures\n\n"
               "What legal question can I help you with today?")