# ============================================
# QUERY - Two Functions: Streaming Answer + Evidence
# ============================================

import time
import torch
import os
import threading
from typing import List, Dict, Any, Generator
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from langchain_nebius import ChatNebius
from langchain_core.prompts import PromptTemplate

from app.config import settings

# ============================================
# CONFIGURATION
# ============================================

QDRANT_LOCATION = settings.QDRANT_LOCATION
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
RETRIEVAL_K = 20
TOP_K = 10
USE_RERANKER = True
EVIDENCE_CHUNKS_TO_PROCESS = 8

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TEMPERATURE = 0

FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
FAST_LLM_TEMPERATURE = 0

# ============================================

class DocumentQuery:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retriever = None
        self.vector_store = None
        self.client = None
        self.llm = None
        self.prompt = None
        self.fast_llm = None
        self.reranker = None
        self.initialized = False
        
        print("="*60)
        print("🚀 QUERY SYSTEM - TWO FUNCTIONS")
        print("="*60)
        print(f"📌 Device: {self.device.upper()}")
        print("="*60 + "\n")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================
    # CONNECTION & SETUP (Same as before)
    # ============================================
    
    def connect_to_database(self):
        """Connect to existing database."""
        print("🔗 Connecting to existing database...")
        try:
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
            
            self.client = QdrantClient(path=QDRANT_LOCATION)
            
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if COLLECTION_NAME not in collection_names:
                print(f"❌ Collection '{COLLECTION_NAME}' not found!")
                return False
            
            print("🧠 Loading embedding model...")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': self.device},
                encode_kwargs={'device': self.device, 'batch_size': 64, 'normalize_embeddings': True}
            )
            
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
            )
            print("✅ Connected to database\n")
            return True
            
        except Exception as e:
            print(f"⚠️ Connection failed: {e}")
            return False
    
    def create_retriever(self):
        """Create retriever with reranking."""
        if not self.vector_store:
            raise ValueError("No database available.")
        
        print("🔍 Creating retriever with reranking...")
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        if USE_RERANKER:
            try:
                print(f"🔄 Loading reranker on {self.device.upper()}...")
                self.reranker = CrossEncoder(
                    'BAAI/bge-reranker-large',
                    device=self.device,
                    num_labels=1,
                    max_length=512
                )
                print("✅ Reranker loaded")
                
                class RerankingRetriever:
                    def __init__(self, base_retriever, reranker, top_k=TOP_K, batch_size=128):
                        self.base_retriever = base_retriever
                        self.reranker = reranker
                        self.top_k = top_k
                        self.batch_size = batch_size
                    
                    def invoke(self, query):
                        initial_docs = self.base_retriever.invoke(query)
                        if not initial_docs:
                            return []
                        pairs = [[query, doc.page_content] for doc in initial_docs]
                        scores = self.reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
                        doc_score_pairs = list(zip(initial_docs, scores))
                        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                        return [doc for doc, score in doc_score_pairs[:self.top_k]]
                
                self.retriever = RerankingRetriever(
                    base_retriever=base_retriever,
                    reranker=self.reranker,
                    top_k=TOP_K
                )
                print(f"✅ Retriever ready: Retrieves {RETRIEVAL_K}, reranks to top {TOP_K}")
                
            except Exception as e:
                print(f"⚠️ Reranker failed: {e}")
                self.retriever = base_retriever
                print(f"✅ Retriever ready WITHOUT reranking")
        else:
            self.retriever = base_retriever
            print(f"✅ Retriever ready WITHOUT reranking")
        
        print()
        return self.retriever
    
    def setup_llm(self):
        """Setup LLM and prompts"""
        print("🤖 Setting up LLMs...")
        
        self.llm = ChatNebius(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            streaming=True
        )
        
        self.fast_llm = ChatNebius(
            model=FAST_LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=FAST_LLM_TEMPERATURE,
            streaming=False
        )
        
        prompt_template = """
Human: You are an AI assistant specialized in the given file. 
Provide accurate and concise answers based on the PDF file.
Use the following pieces of information to provide a concise answer to the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Cite specific sections (like "Section 100.1") when possible.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        print("✅ LLMs ready\n")
    
    def initialize(self):
        """Initialize the query system."""
        if not self.connect_to_database():
            print("\n❌ Could not connect to database!")
            return False
        
        self.create_retriever()
        self.setup_llm()
        self.initialized = True
        
        print("="*60)
        print("✅ QUERY SYSTEM READY!")
        print(f"📌 Main Model: {LLM_MODEL}")
        print(f"📌 Fast Model: {FAST_LLM_MODEL}")
        print("="*60 + "\n")
        
        return True
    
    # ============================================
    # FUNCTION 1: GET ANSWER AS STREAM
    # ============================================
    
    def get_answer_stream(self, question: str) -> Generator[str, None, None]:
        """
        FUNCTION 1: Get answer as a stream.
        
        Yields answer chunks in real-time.
        User sees text appearing word by word.
        
        Usage:
            for chunk in query_system.get_answer_stream("What is X?"):
                print(chunk, end="", flush=True)
        """
        if not self.initialized:
            yield "❌ System not initialized."
            return
        
        print("\n" + "="*60)
        print(f"📝 Streaming Answer for: {question}")
        print("="*60 + "\n")
        
        # Step 1: Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Step 2: Generate answer with streaming
        context = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        print("🤖 Answer: ", end="", flush=True)
        
        # Stream the answer
        for chunk in self.llm.stream(formatted_prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                yield chunk.content  # ← Send to caller
        
        print("\n\n✅ Answer complete!")
        print("="*60 + "\n")
    
    # ============================================
    # FUNCTION 2: GET EVIDENCE FOR HIGHLIGHTING
    # ============================================
    
    def get_evidence(self, question: str) -> str:
        """
        FUNCTION 2: Get supporting evidence for highlighting.
        
        Returns evidence text only (no streaming).
        Used for PDF highlighting or internal processing.
        
        Usage:
            evidence = query_system.get_evidence("What is X?")
            # Use evidence for highlighting PDF
        """
        if not self.initialized:
            return "❌ System not initialized."
        
        print("\n" + "="*60)
        print(f"📋 Extracting Evidence for: {question}")
        print("="*60 + "\n")
        
        # Step 1: Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Step 2: Extract evidence
        evidence = self.extract_evidence_from_question(question, docs)
        
        print(f"✅ Evidence extracted! ({len(evidence)} characters)")
        print("="*60 + "\n")
        
        return evidence
    
    # ============================================
    # HELPER: Extract Evidence (Same as before)
    # ============================================
    
    def extract_evidence_from_question(self, question: str, chunks: List[Any]) -> str:
        """Extract evidence using QUESTION + CHUNKS."""
        if not chunks or not question:
            return ""
        
        print(f"   Using {FAST_LLM_MODEL}")
        print(f"   Processing {min(len(chunks), EVIDENCE_CHUNKS_TO_PROCESS)} chunks")
        
        # Prepare chunks
        chunk_texts = []
        for idx, doc in enumerate(chunks[:EVIDENCE_CHUNKS_TO_PROCESS]):
            chunk_id = idx + 1
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            
            chunk_texts.append(f"""
[CHUNK {chunk_id}] - Source: {source}, Page: {page}
{content}
""")
        
        all_chunks = "\n\n".join(chunk_texts)
        
        prompt = f"""
Extract supporting evidence from these chunks that directly answers the question.

QUESTION:
{question}

CHUNKS:
{all_chunks}

INSTRUCTIONS:
Extract only complete sentences that directly support the question.

Copy every extracted sentence verbatim from the retrieved chunks.

Do NOT modify anything: no insertions, deletions, replacements, corrections, paraphrasing, summarizing, completing, reordering, or changes to any character (letters, numbers, spaces, punctuation, symbols, capitalization, line breaks).

Do NOT add any text not present in the retrieved chunks.

Do NOT extract incomplete sentences (those that begin or end mid-chunk).

Do NOT correct any errors.

Start each extracted sentence on a new line.

Output ONLY the extracted text.

EVIDENCE:"""
        
        try:
            result = self.fast_llm.invoke(prompt)
            evidence = result.content.strip()
            
            if evidence:
                print("   ✅ Evidence extracted successfully")
                return evidence
            else:
                print("   ⚠️ No evidence found")
                return ""
            
        except Exception as e:
            print(f"   ⚠️ Extraction failed: {e}")
            return ""
    
    # ============================================
    # COMBINED: Get Both Answer + Evidence (For FastAPI)
    # ============================================
    
    def get_answer_and_evidence(self, question: str) -> Dict[str, Any]:
        """
        Combined function for FastAPI.
        Returns both answer (full) and evidence.
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        # Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Run in parallel
        evidence_result = None
        evidence_time = 0
        
        def extract_evidence_background():
            nonlocal evidence_result, evidence_time
            import time
            evidence_start = time.time()
            evidence_result = self.extract_evidence_from_question(question, docs)
            evidence_time = time.time() - evidence_start
        
        evidence_thread = threading.Thread(target=extract_evidence_background)
        evidence_thread.start()
        
        # Generate answer
        context = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(formatted_prompt)
        
        # Wait for evidence
        evidence_thread.join(timeout=30)
        
        return {
            'answer': answer.content,
            'evidence': evidence_result or "No relevant evidence found.",
            'chunks': docs[:TOP_K]  # For reference
        }

# ============================================
# SINGLETON INSTANCE
# ============================================

query_system = None

def get_query_system():
    """Get or initialize the query system"""
    global query_system
    if query_system is None:
        query_system = DocumentQuery()
        if not query_system.initialize():
            return None
    return query_system

# ============================================
# MAIN - DEMONSTRATE THE TWO FUNCTIONS
# ============================================

if __name__ == "__main__":
    # Initialize
    system = get_query_system()
    if not system:
        exit(1)
    
    question = "What is the minimum wage in Canada?"
    
    # ============================================
    # FUNCTION 1: Streaming Answer
    # ============================================
    print("\n" + "="*60)
    print("EXAMPLE 1: GET ANSWER AS STREAM")
    print("="*60)
    print("User sees text appearing word by word:")
    print()
    
    for chunk in system.get_answer_stream(question):
        # In real app, this would be sent to frontend
        pass  # Already printed inside get_answer_stream
    
    # ============================================
    # FUNCTION 2: Get Evidence for Highlighting
    # ============================================
    print("\n" + "="*60)
    print("EXAMPLE 2: GET EVIDENCE FOR HIGHLIGHTING")
    print("="*60)
    print("This will be used for PDF highlighting:")
    print()
    
    evidence = system.get_evidence(question)
    
    print("\n📋 EVIDENCE TEXT:")
    print("-"*60)
    print(evidence[:500] + "..." if len(evidence) > 500 else evidence)
    print("-"*60)
    
    # You can now use this evidence for PDF highlighting
    print("\n✅ Evidence saved for highlighting")
    print("   You can now highlight this in the PDF")
