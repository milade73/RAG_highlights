# ============================================
# QUERY - TRUE PARALLEL with Question-Based Evidence
# ============================================

import time
import torch
import os
import sys
import json
import concurrent.futures
import threading
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from langchain_nebius import ChatNebius
from langchain_core.prompts import PromptTemplate
import re
from typing import List, Tuple, Dict, Any, Optional
from queue import Queue

# ============================================
# CONFIGURATION - TRUE PARALLEL
# ============================================

# Qdrant Settings
QDRANT_LOCATION = "./qdrant_data"
COLLECTION_NAME = "labour_code"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Retrieval Settings
RETRIEVAL_K = 20
TOP_K = 10
USE_RERANKER = True
RERANKER_BATCH_SIZE = 128

# LLM Settings - Main model for answers (Quality)
LLM_MODEL = "openai/gpt-oss-120b" 
LLM_API_KEY = r"v1.CmMKHHN0YXRpY2tleS1lMDB0MGZkcnlmd3gzOXRkc2ESIXNlcnZpY2VhY2NvdW50LWUwMHQzZXdidGJ0N3kyYnNxbjIMCPutm9EGELLIkKwBOgsI-rCznAcQwMjnIkACWgNlMDA.AAAAAAAAAAG_9-wUVIErZ1TcN7Q_805hGx0ODQmPbWY1eisvtbpkxoIGVfaqm_ngNRCZoJFs8U0nbH2pTwCtq50N1xgC0r0D"
LLM_TEMPERATURE = 0

# FAST LLM for evidence extraction - Uses QUESTION (not answer)
FAST_LLM_MODEL =  "Qwen/Qwen3-235B-A22B-Instruct-2507"
FAST_LLM_TEMPERATURE = 0

# Evidence Extraction Settings
EVIDENCE_CHUNKS_TO_PROCESS = 8

# Output Settings
OUTPUT_DIR = "./query_results"
FILTERED_OUTPUT_DIR = "./filtered_results"

# ============================================

class DocumentQuery:
    """Query existing database with TRUE PARALLEL processing"""
    
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
        self.query_count = 0
        
        # Timing tracking
        self.timings = {
            'retrieval': 0.0,
            'answer_generation': 0.0,
            'evidence_extraction': 0.0,
            'total': 0.0
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FILTERED_OUTPUT_DIR, exist_ok=True)
        
        print("="*60)
        print("🚀 DOCUMENT QUERY SYSTEM - TRUE PARALLEL")
        print("="*60)
        print(f"📌 Device: {self.device.upper()}")
        print(f"📌 Location: {QDRANT_LOCATION}")
        print(f"📌 Collection: {COLLECTION_NAME}")
        print(f"📌 Main Model: {LLM_MODEL}")
        print(f"📌 Fast Model: {FAST_LLM_MODEL}")
        print(f"📌 Evidence Chunks: {EVIDENCE_CHUNKS_TO_PROCESS}")
        print(f"📌 Mode: TRUE PARALLEL (Question + Chunks for evidence)")
        print("="*60 + "\n")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ GPU memory cleared\n")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into complete sentences."""
        lines = text.split('\n')
        sentences = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                is_french = re.search(r'[éèêàç]', next_line) or any(
                    word in next_line.lower() for word in ['des', 'du', 'de', 'la', 'le', 'les', 'l\'', 'à', 'dans', 'pour', 'sur']
                )
                is_english = not re.search(r'[éèêàç]', line)
                
                if is_english and is_french and len(line) > 10 and len(next_line) > 10:
                    sentences.append(f"{line}\n{next_line}")
                    i += 2
                    continue
            
            if re.search(r'\d+\.?\d*\s+', line):
                bilingual_text = [line]
                j = i + 1
                while j < len(lines) and (
                    re.search(r'[éèêàç]', lines[j]) or 
                    len(lines[j].strip()) < 50 or
                    lines[j].strip().startswith(('L.R.', 'R.S.', 'L.R.', 'R.S.'))
                ):
                    if lines[j].strip():
                        bilingual_text.append(lines[j].strip())
                    j += 1
                if len(bilingual_text) > 1:
                    sentences.append('\n'.join(bilingual_text))
                    i = j
                    continue
            
            if line and len(line) > 5:
                sentences.append(line)
            i += 1
        
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        return sentences
    
    def extract_evidence_from_question(self, question: str, chunks: List[Any]) -> str:
        """
        Extract evidence using QUESTION + CHUNKS (NO answer needed!)
        This can run in TRUE PARALLEL with answer generation.
        """
        if not chunks or not question:
            return ""
        
        print(f"\n📋 Extracting evidence from QUESTION + CHUNKS (parallel)...")
        print(f"   Model: {FAST_LLM_MODEL}")
        print(f"   Chunks: {min(len(chunks), EVIDENCE_CHUNKS_TO_PROCESS)}")
        print("="*60)
        
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
        
        # PROMPT uses QUESTION (not answer) - TRUE PARALLEL!
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

Do NOT add any text not present in the retrieved chunks, including abbreviations, expansions, explanations, parentheses, brackets, or any other additional content.

Do NOT merge multiple sentences or split a sentence into smaller parts.

Do NOT extract incomplete sentences (those that begin or end mid-chunk).

Do NOT correct any errors (grammatical, spelling, punctuation, or any other). Copy errors exactly as they appear.

Start each extracted sentence on a new line. Do NOT put multiple sentences on the same line.

Group related sentences into paragraphs with a blank line between groups. Do NOT add blank lines within a group.

Extract each unique sentence only once. Do NOT output multiple versions, variations, paraphrases, or alternative wordings of the same sentence.

Output ONLY the extracted text. Do NOT add headings, explanations, commentary, transitions, bullet points, numbering, or any other text.

CRITICAL: Extract ALL complete sentences from the retrieved chunks that directly support the question. Do NOT omit, skip, or leave out any relevant sentence. If a sentence is complete and supports the question, you MUST include it. Do NOT truncate, shorten, or partially extract a sentence. The full sentence must be copied from beginning to end.

CRITICAL:  SKIP FORMULAS AND EQUATIONS: Do NOT extract mathematical formulas, equations, chemical structures, or any special mathematical notation. Extract ONLY the natural language text (complete sentences) that directly support the question. Skip any sentence that consists primarily of or contains embedded mathematical expressions, equations, or formula notation. Extract only the surrounding explanatory text without the formulas.

CRITICAL: If a sentence is identified as supporting evidence (in English or French), return the complete paragraph containing that sentence exactly as it appears in the original document, preserving its original language, formatting, and line breaks without translation, summarization, or modification.
EVIDENCE:"""
        
        try:
            result = self.fast_llm.invoke(prompt)
            evidence = result.content.strip()
            
            if evidence:
                print(f"   ✅ Evidence extracted (parallel with answer)")
                return evidence
            else:
                print("   ⚠️ No evidence found")
                return ""
            
        except Exception as e:
            print(f"   ⚠️ Extraction failed: {e}")
            return ""
    
    def generate_answer(self, question: str, chunks: List[Any]) -> tuple:
        """Generate final answer using main LLM"""
        context = "\n\n".join(doc.page_content for doc in chunks)
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        print("\n🤖 Final Answer: ", end="", flush=True)
        
        start_time = time.time()
        full_response = ""
        for chunk in self.llm.stream(formatted_prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        generation_time = time.time() - start_time
        print()
        
        return full_response, generation_time
    
    def save_evidence_for_highlighting(self, question: str, answer: str, 
                                      evidence_text: str, chunks: List[Any],
                                      retrieval_time: float, generation_time: float, 
                                      evidence_time: float) -> Optional[str]:
        """Save the extracted evidence for highlighting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.query_count += 1
        
        os.makedirs(FILTERED_OUTPUT_DIR, exist_ok=True)
        
        filename = f"evidence_for_highlighting_{self.query_count:04d}_{timestamp}.txt"
        filepath = os.path.join(FILTERED_OUTPUT_DIR, filename)
        
        total_chunks = len(chunks)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write(f"📌 EVIDENCE FOR HIGHLIGHTING\n")
                f.write("="*60 + "\n")
                f.write(f"Query: {question}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Chunks used: {total_chunks}\n")
                f.write(f"Evidence model: {FAST_LLM_MODEL}\n")
                f.write(f"Evidence method: QUESTION + CHUNKS (TRUE PARALLEL)\n")
                f.write(f"Evidence chunks: {EVIDENCE_CHUNKS_TO_PROCESS}\n")
                f.write(f"\n{'-'*60}\n")
                f.write(f"EXTRACTION STATISTICS:\n")
                f.write(f"  • Retrieval time: {retrieval_time:.2f}s\n")
                f.write(f"  • Answer generation: {generation_time:.2f}s\n")
                f.write(f"  • Evidence extraction: {evidence_time:.2f}s\n")
                f.write(f"  • Total time: {retrieval_time + generation_time + evidence_time:.2f}s\n")
                f.write(f"{'-'*60}\n\n")
                
                f.write("="*60 + "\n")
                f.write("📝 FINAL ANSWER:\n")
                f.write("="*60 + "\n")
                f.write(answer)
                f.write("\n\n")
                
                f.write("="*60 + "\n")
                f.write("📄 SUPPORTING EVIDENCE (For Highlighting)\n")
                f.write("="*60 + "\n")
                f.write(evidence_text if evidence_text else "No relevant evidence found.")
                f.write("\n\n")
                
                f.write("="*60 + "\n")
                f.write("📋 ALL SOURCE CHUNKS\n")
                f.write("="*60 + "\n\n")
                
                for idx, doc in enumerate(chunks):
                    chunk_id = idx + 1
                    f.write(f"--- Chunk {chunk_id} ---\n")
                    f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
                    f.write(f"Page: {doc.metadata.get('page', '?')}\n")
                    f.write(f"\n{doc.page_content}\n")
                    f.write(f"\n{'-'*60}\n\n")
            
            return filepath
            
        except Exception as e:
            return None
    
    def save_all_chunks(self, question: str, docs: List[Any], response: str, 
                       retrieval_time: float, generation_time: float) -> str:
        """Save ALL chunks (original retrieval results)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.query_count += 1
        
        filename = f"all_chunks_{self.query_count:04d}_{timestamp}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"ALL CHUNKS - Original Retrieval Results\n")
                f.write(f"{'='*60}\n")
                f.write(f"Query: {question}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Total Chunks: {len(docs)}\n")
                f.write(f"\n{'='*60}\n\n")
                
                for i, doc in enumerate(docs, 1):
                    f.write(f"--- Chunk {i} ---\n")
                    f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
                    f.write(f"Page: {doc.metadata.get('page', '?')}\n")
                    f.write(f"\n{doc.page_content}\n")
                    f.write(f"\n{'-'*60}\n\n")
                
                f.write(f"{'='*60}\n")
                f.write(f"FINAL ANSWER:\n")
                f.write(f"{'='*60}\n")
                f.write(response)
                f.write("\n")
            
            return filepath
            
        except Exception as e:
            return None
    
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
                    def __init__(self, base_retriever, reranker, top_k=TOP_K, batch_size=RERANKER_BATCH_SIZE):
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
                    top_k=TOP_K,
                    batch_size=RERANKER_BATCH_SIZE
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
        
        # Main LLM for answering (Quality)
        print(f"   📌 Main Answer Model: {LLM_MODEL}")
        self.llm = ChatNebius(
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            streaming=True
        )
        
        # FAST LLM for evidence extraction (Speed - uses QUESTION)
        print(f"   📌 Fast Evidence Model: {FAST_LLM_MODEL}")
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
        print("✅ QUERY SYSTEM READY - TRUE PARALLEL!")
        print(f"📌 Main Model: {LLM_MODEL} (Answer Generation)")
        print(f"📌 Fast Model: {FAST_LLM_MODEL} (Evidence Extraction)")
        print(f"📌 Evidence Chunks: {EVIDENCE_CHUNKS_TO_PROCESS}")
        print(f"📌 Mode: TRUE PARALLEL (Question + Chunks)")
        print(f"📌 Will save 2 files:")
        print(f"   1. ALL {TOP_K} chunks (no filtering)")
        print(f"   2. Supporting evidence for highlighting")
        print("="*60 + "\n")
        
        return True
    
    def ask(self, question, show_metadata=True, show_context=False, clean_results=True):
        """Ask a question with TRUE parallel processing."""
        if not self.initialized:
            print("❌ System not initialized.")
            return None
        
        # Reset timing
        self.timings = {k: 0.0 for k in self.timings}
        total_start_time = time.time()
        
        print("="*60)
        print(f"📝 Question: {question}")
        print("="*60)
        
        # ============================================
        # STEP 1: Retrieve and rerank
        # ============================================
        start_time = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - start_time
        self.timings['retrieval'] = retrieval_time
        
        # ============================================
        # STEP 2: TRUE PARALLEL - Answer + Evidence
        # ============================================
        
        print("\n" + "="*60)
        print("⚡ RUNNING ANSWER & EVIDENCE IN PARALLEL")
        print("="*60)
        
        evidence_result = None
        evidence_time = 0
        answer = ""
        generation_time = 0
        
        # Start evidence extraction in background thread (uses QUESTION + CHUNKS)
        def extract_evidence_background():
            nonlocal evidence_result, evidence_time
            evidence_start = time.time()
            evidence_result = self.extract_evidence_from_question(question, docs)
            evidence_time = time.time() - evidence_start
        
        evidence_thread = threading.Thread(target=extract_evidence_background)
        evidence_thread.start()
        
        # Generate answer in main thread
        answer, generation_time = self.generate_answer(question, docs)
        self.timings['answer_generation'] = generation_time
        
        # Wait for evidence thread to complete
        evidence_thread.join(timeout=30)
        self.timings['evidence_extraction'] = evidence_time
        
        print("\n" + "="*60)
        print(f"✅ Both tasks completed!")
        print(f"   Answer: {generation_time:.2f}s | Evidence: {evidence_time:.2f}s")
        print("="*60 + "\n")
        
        # ============================================
        # STEP 3: Save evidence and chunks
        # ============================================
        evidence_file = None
        
        if clean_results and answer and docs and evidence_result:
            evidence_file = self.save_evidence_for_highlighting(
                question, answer, evidence_result, docs,
                retrieval_time, generation_time, evidence_time
            )
        
        # ============================================
        # STEP 4: Save all chunks
        # ============================================
        all_chunks_file = self.save_all_chunks(
            question, docs, answer, retrieval_time, generation_time
        )
        
        # Calculate total time
        total_time = time.time() - total_start_time
        self.timings['total'] = total_time
        
        # Print summary
        print("\n" + "="*60)
        print("✅ DONE! Two files saved:")
        if all_chunks_file:
            print(f"   1. ALL chunks: {all_chunks_file}")
        if evidence_file:
            print(f"   2. Supporting evidence: {evidence_file}")
        
        # Timing summary
        print("\n" + "="*60)
        print("⏱️  TIMING SUMMARY (TRUE PARALLEL)")
        print("="*60)
        print(f"   📥 Retrieval:              {self.timings['retrieval']:.3f}s")
        print(f"   🤖 Answer Generation:      {self.timings['answer_generation']:.3f}s")
        print(f"   📋 Evidence Extraction:    {self.timings['evidence_extraction']:.3f}s")
        print("-"*60)
        print(f"   ⏱️  TOTAL TIME:            {self.timings['total']:.3f}s")
        print("="*60)
        print("💡 Evidence extracted using QUESTION + CHUNKS")
        print("📌 TRUE PARALLEL - Evidence ran while answer was generated!")
        print("="*60 + "\n")
        
        return answer

    # ============================================
    # NEW FUNCTION 1: GET ANSWER AS STREAM
    # ============================================
    
    def get_answer_stream(self, question: str, save: bool = False):
        """
        Get answer as a stream - returns chunks in real-time.
        Uses your existing generate_answer() but yields instead of printing.
        
        Args:
            question: The question to ask
            save: If True, saves filtered answer to file. If False, only returns.
        
        Yields:
            str: Chunks of the answer
        """
        if not self.initialized:
            yield "❌ System not initialized."
            return
        
        print("="*60)
        print(f"📝 Question: {question}")
        print("="*60)
        
        # Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Generate answer with streaming
        context = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        print("\n🤖 Final Answer: ", end="", flush=True)
        
        full_response = ""
        for chunk in self.llm.stream(formatted_prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
                yield chunk.content
        
        print()
        print("="*60 + "\n")
        
        # ============================================
        # OPTIONAL: SAVE FILTERED RESULTS (Answer Only)
        # ============================================
        if save and full_response:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_count += 1
            
            filename = f"filtered_answer_{self.query_count:04d}_{timestamp}.txt"
            filepath = os.path.join(FILTERED_OUTPUT_DIR, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("📌 FILTERED ANSWER\n")
                    f.write("="*60 + "\n")
                    f.write(f"Query: {question}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Model: {LLM_MODEL}\n")
                    f.write(f"\n{'-'*60}\n\n")
                    f.write(full_response)
                    f.write("\n")
                
                print(f"💾 Filtered answer saved to: {filepath}")
                
            except Exception as e:
                print(f"⚠️ Failed to save answer: {e}")
    
    # ============================================
    # NEW FUNCTION 2: GET EVIDENCE ONLY
    # ============================================
    
    def get_evidence(self, question: str, save: bool = False) -> str:
        """
        Get supporting evidence only - returns extracted sentences.
        Uses your existing extract_evidence_from_question().
        Returns exactly what the LLM returns - no modifications.
        
        Args:
            question: The question to ask
            save: If True, saves filtered evidence to file. If False, only returns.
        
        Returns:
            str: The evidence text
        """
        if not self.initialized:
            return "❌ System not initialized."
        
        print("="*60)
        print(f"📝 Question: {question}")
        print("="*60)
        
        # Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Extract evidence using your existing method
        evidence = self.extract_evidence_from_question(question, docs)
        
        # ============================================
        # OPTIONAL: SAVE FILTERED RESULTS (Evidence Only)
        # ============================================
        if save and evidence:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_count += 1
            
            filename = f"filtered_evidence_{self.query_count:04d}_{timestamp}.txt"
            filepath = os.path.join(FILTERED_OUTPUT_DIR, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("📌 FILTERED EVIDENCE (For Highlighting)\n")
                    f.write("="*60 + "\n")
                    f.write(f"Query: {question}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Evidence Model: {FAST_LLM_MODEL}\n")
                    f.write(f"\n{'-'*60}\n\n")
                    f.write(evidence)
                    f.write("\n")
                
                print(f"💾 Filtered evidence saved to: {filepath}")
                
            except Exception as e:
                print(f"⚠️ Failed to save evidence: {e}")
        
        print("="*60 + "\n")
        
        return evidence

    # ============================================
    # NEW FUNCTION 3: GET BOTH (Evidence + Answer)
    # ============================================
    
    def get_evidence_and_answer(self, question: str, save: bool = False) -> Dict[str, Any]:
        """
        Get both evidence and answer in one call.
        
        Args:
            question: The question to ask
            save: If True, saves filtered results to file
        
        Returns:
            Dict with 'answer' and 'evidence' keys
        """
        if not self.initialized:
            return {'error': 'System not initialized'}
        
        print("="*60)
        print(f"📝 Question: {question}")
        print("="*60)
        
        # Retrieve chunks
        docs = self.retriever.invoke(question)
        
        # Run in parallel
        evidence_result = None
        evidence_time = 0
        
        def extract_evidence_background():
            nonlocal evidence_result, evidence_time
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
        
        result = {
            'answer': answer.content,
            'evidence': evidence_result or "No relevant evidence found."
        }
        
        # ============================================
        # OPTIONAL: SAVE FILTERED RESULTS
        # ============================================
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_count += 1
            
            # Save evidence
            if evidence_result:
                evidence_file = f"filtered_evidence_{self.query_count:04d}_{timestamp}.txt"
                evidence_path = os.path.join(FILTERED_OUTPUT_DIR, evidence_file)
                
                with open(evidence_path, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("📌 FILTERED EVIDENCE\n")
                    f.write("="*60 + "\n")
                    f.write(f"Query: {question}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"\n{'-'*60}\n\n")
                    f.write(evidence_result)
                    f.write("\n")
                
                print(f"💾 Evidence saved to: {evidence_path}")
            
            # Save answer
            if answer.content:
                answer_file = f"filtered_answer_{self.query_count:04d}_{timestamp}.txt"
                answer_path = os.path.join(FILTERED_OUTPUT_DIR, answer_file)
                
                with open(answer_path, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("📌 FILTERED ANSWER\n")
                    f.write("="*60 + "\n")
                    f.write(f"Query: {question}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"\n{'-'*60}\n\n")
                    f.write(answer.content)
                    f.write("\n")
                
                print(f"💾 Answer saved to: {answer_path}")
        
        return result

# ============================================
# INTERACTIVE MODE
# ============================================

def interactive_mode():
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE - TRUE PARALLEL")
    print("="*60)
    print("Type your questions below.")
    print("Type 'quit', 'exit', or 'q' to quit.")
    print("="*60 + "\n")
    
    query_system = DocumentQuery()
    if not query_system.initialize():
        return
    
    while True:
        try:
            question = input("\n🔍 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            query_system.ask(question, show_metadata=True, show_context=False, clean_results=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    elif len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        query_system = DocumentQuery()
        if query_system.initialize():
            query_system.ask(question, show_metadata=True, show_context=False, clean_results=True)
    else:
        interactive_mode()
