
import json
import os
import logging
import re
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pydantic import BaseModel, ValidationError
import streamlit as st
import hashlib
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Disable GPU Faiss warnings
os.environ['FAISS_NO_GPU'] = '1'

class LexaError(Exception):
    """Custom exception for Lexa-specific errors"""
    pass

class DatasetItem(BaseModel):
    """Pydantic model for dataset validation"""
    title: Optional[str] = None
    topic: Optional[str] = None
    name: Optional[str] = None
    content: Optional[str] = None
    definition: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None

class LexaCore:
    def __init__(self):
        """Initialize the Lexa legal AI assistant"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self._validate_environment()
        
        # Model configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model = os.getenv("LLM_MODEL", "mistralai/mixtral-8x7b-instruct")
        self.api_key = self._get_api_key()
        self.similarity_k = int(os.getenv("FAISS_SIMILARITY_K", 3))
        
        # Initialize components
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectorstore = self._load_vectorstore()
            self.llm = self._initialize_llm()
            self.prompt_template = self._get_prompt_template()
            self.query_cache = {}  # Simple in-memory cache for queries
            self.logger.info("LexaCore initialized successfully")
        except Exception as e:
            self.logger.error(f"LexaCore initialization failed: {e}")
            raise LexaError(f"Failed to initialize LexaCore: {str(e)}")

    def _validate_environment(self) -> None:
        """Validate required environment setup"""
        required_files = ["contract_law_dataset.json", "land_law_dataset.json"]
        for file in required_files:
            if not os.path.exists(file):
                raise LexaError(f"Required data file not found: {file}")
        cache_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        if not os.access(os.path.dirname(cache_path) or ".", os.W_OK):
            raise LexaError(f"FAISS index path is not writable: {cache_path}")

    def _get_api_key(self) -> str:
        """Retrieve API key from Streamlit secrets or environment"""
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise LexaError("API key not found in configuration")
            return api_key
        except Exception as e:
            self.logger.error("API key retrieval failed: [REDACTED]")
            raise LexaError("API key configuration missing")

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM client with retry logic"""
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def init_llm():
            return ChatOpenAI(
                model=self.llm_model,
                openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                openai_api_key=self.api_key,
                temperature=0.3,
                max_tokens=2000
            )
        return init_llm()

    def _load_vectorstore(self) -> FAISS:
        """Load or create the FAISS vectorstore with validation"""
        cache_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        
        try:
            if os.path.exists(cache_path):
                self.logger.info("Loading existing FAISS index")
                # Validate index file integrity
                expected_checksum = os.getenv("FAISS_INDEX_CHECKSUM")
                if expected_checksum:
                    with open(cache_path, "rb") as f:
                        actual_checksum = hashlib.sha256(f.read()).hexdigest()
                    if actual_checksum != expected_checksum:
                        raise LexaError("FAISS index checksum mismatch")
                return FAISS.load_local(
                    cache_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=False
                )

            self.logger.info("Creating new FAISS index")
            documents = self._load_documents()
            
            if not documents:
                raise LexaError("No legal documents loaded")

            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.save_local(cache_path)
            with open(cache_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
                self.logger.info(f"FAISS index created with checksum: {checksum}")
            return vectorstore

        except Exception as e:
            self.logger.error(f"Vectorstore initialization failed: {e}")
            raise LexaError("Failed to initialize legal knowledge base")

    def _load_documents(self) -> List[Document]:
        """Load and process legal documents with validation"""
        datasets = ['contract_law_dataset.json', 'land_law_dataset.json']
        documents = []
        
        for dataset in datasets:
            try:
                with open(dataset, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    try:
                        validated_item = DatasetItem(**item)
                        content = self._extract_content(validated_item)
                        title = self._extract_title(validated_item)
                        metadata = self._extract_metadata(validated_item, dataset)
                        
                        documents.append(Document(
                            page_content=f"{title}\n\n{content}",
                            metadata=metadata
                        ))
                    except ValidationError as ve:
                        self.logger.warning(f"Invalid item in {dataset}: {ve}")
                        continue
                
                self.logger.debug(f"Loaded {len(documents)} documents from {dataset}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load {dataset}: {e}")
                continue
                
        if not documents:
            raise LexaError("No valid legal documents loaded from any dataset")
            
        return documents

    def _extract_content(self, item: DatasetItem) -> str:
        """Extract content from document item"""
        return (item.content or item.definition 
                or item.text or item.summary or "")

    def _extract_title(self, item: DatasetItem) -> str:
        """Extract title from document item"""
        return (item.title or item.topic 
                or item.name or "Untitled")

    def _extract_metadata(self, item: DatasetItem, source: str) -> Dict[str, Any]:
        """Extract and clean metadata"""
        metadata = {
            'source': source,
            'title': self._extract_title(item),
            **{k: v for k, v in item.dict().items() 
              if k not in ['content', 'definition', 'text', 'summary', 'title', 'topic', 'name']
              and v is not None}
        }
        return metadata

    def _get_prompt_template(self) -> PromptTemplate:
        """Return the structured legal analysis prompt template"""
        try:
            template = """You are Lexa, a Nigerian legal assistant. Provide thorough legal analysis based only on provided context and Nigerian law:

1. **Legal Issue**: Identify the core legal question
2. **Applicable Law**: Cite relevant Nigerian statutes/cases from context
3. **Analysis**: Apply law to the facts
4. **Conclusion**: Practical guidance

Context: {context}

Question: {query}

Respond in clear, professional language suitable for legal professionals. Do not invent laws or cases."""
            prompt = PromptTemplate(
                input_variables=["context", "query"],
                template=template
            )
            self.logger.debug("Prompt template created successfully")
            return prompt
        except Exception as e:
            self.logger.error(f"Failed to create prompt template: {e}")
            raise LexaError(f"Prompt template initialization failed: {str(e)}")

    def process_query(self, query: str) -> str:
        """Process a legal query and return response"""
        try:
            query = query.strip()
            if not query:
                return "Please ask a legal question about Nigerian law."
                
            if self._is_greeting(query):
                return self._get_greeting_response()

            # Check cache
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self.query_cache:
                self.logger.debug(f"Cache hit for query: {query[:50]}...")
                return self.query_cache[cache_key]

            self.logger.info(f"Processing query: {query[:50]}...")
            docs = self.vectorstore.similarity_search(query, k=self.similarity_k)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            prompt = self.prompt_template.format(
                context=context,
                query=query
            )
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            self.query_cache[cache_key] = response_text
            self.logger.debug(f"Query processed in {response.response_metadata.get('processing_time', 'unknown')} seconds")
            return response_text
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise LexaError("Sorry, I encountered an error. Please rephrase your question.")

    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting"""
        greeting_pattern = r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|sup|yo)\b"
        return bool(re.match(greeting_pattern, text.lower(), re.IGNORECASE))

    def _get_greeting_response(self) -> str:
        """Return appropriate greeting response"""
        hour = datetime.now().hour
        greeting = "Hello"
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"
        return (f"{greeting}! I'm Lexa, your Nigerian legal assistant. "
               "I can help with:\n\n"
               "- Contract Law\n- Land Law\n- Legal Rights\n- Court Procedures\n\n"
               "What legal question can I help you with today?")

    def reset(self) -> None:
        """Reset LexaCore state"""
        self.query_cache.clear()
        self.logger.info("LexaCore state reset")