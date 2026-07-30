import time
import os
import shutil
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Import settings
from app.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SentenceBasedChunker:
    """Chunker that respects sentence boundaries."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.math_symbols = {
            'ω': 'ω', 'Ω': 'Ω', 'π': 'π', 'Σ': 'Σ',
            'β': 'β', 'γ': 'γ', 'α': 'α', 'δ': 'δ',
            'θ': 'θ', 'λ': 'λ', 'μ': 'μ', 'σ': 'σ',
            'τ': 'τ', 'φ': 'φ', 'ψ': 'ψ', 'ε': 'ε'
        }
    
    def protect_math_expressions(self, text: str) -> str:
        """Protect mathematical expressions from being split."""
        text = re.sub(r'\$\$[^$]+\$\$', lambda m: f"«MATH»{m.group(0)}«/MATH»", text)
        text = re.sub(r'\$[^$]+\$', lambda m: f"«MATH»{m.group(0)}«/MATH»", text)
        for symbol in self.math_symbols:
            text = text.replace(symbol, f"«GREEK»{symbol}«/GREEK»")
        return text
    
    def restore_protected_content(self, text: str) -> str:
        """Restore protected mathematical expressions."""
        text = re.sub(r'«MATH»(.*?)«/MATH»', r'\1', text)
        for symbol in self.math_symbols:
            text = text.replace(f"«GREEK»{symbol}«/GREEK»", symbol)
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into complete sentences."""
        text = self.protect_math_expressions(text)
        parts = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = []
        for part in parts:
            if part and len(part) > 0:
                sentences.append(part.strip())
        
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                sent = self.restore_protected_content(sent)
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap between chunks."""
        if not sentences:
            return []
        
        overlap_sentences = []
        overlap_size = 0
        
        for sentence in reversed(sentences):
            sent_len = len(sentence)
            if overlap_size + sent_len <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += sent_len
            else:
                break
        
        return overlap_sentences
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a document with proper sentence boundaries."""
        text = document.page_content
        metadata = document.metadata
        
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return [document]
        
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_size + sentence_length > self.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_type'] = 'sentence_based'
                chunk_metadata['sentence_count'] = len(current_chunk_sentences)
                chunk_metadata['chunk_size'] = len(chunk_text)
                chunk_metadata['is_complete'] = True
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
                
                overlap_sentences = self.get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences.copy()
                current_size = sum(len(s) for s in overlap_sentences)
            
            current_chunk_sentences.append(sentence)
            current_size += sentence_length
        
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_type'] = 'sentence_based'
            chunk_metadata['sentence_count'] = len(current_chunk_sentences)
            chunk_metadata['chunk_size'] = len(chunk_text)
            chunk_metadata['is_complete'] = True
            
            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class DocumentIngester:
    """Ingest documents into Qdrant - CPU-ONLY Version"""
    
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path
        self.device = "cpu"
        self.vector_store = None
        self.client = None
        self.chunker = SentenceBasedChunker()
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TORCH_DEVICE"] = "cpu"
    
    def preserve_text(self, text: str) -> str:
        """Preserve text exactly as-is with all Unicode characters"""
        if isinstance(text, str):
            return text
        return str(text)
    
    def clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata but preserve Unicode characters"""
        cleaned = {}
        for key, value in metadata.items():
            if key == 'page':
                try:
                    cleaned[key] = int(value) if value else 0
                except (ValueError, TypeError):
                    cleaned[key] = 0
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def clear_existing_db(self, location: str = None):
        """Clear existing database with backup"""
        location = location or settings.QDRANT_LOCATION
        if os.path.exists(location):
            logger.info(f"Removing existing database at {location}...")
            backup_path = f"{location}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copytree(location, backup_path)
                logger.info(f"Backup created at: {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
            shutil.rmtree(location)
            logger.info("Database removed")
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and return documents"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        start_time = time.time()
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages in {time.time() - start_time:.2f}s")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with sentence-based chunking"""
        logger.info("Creating sentence-based chunks...")
        start_time = time.time()
        
        chunks = self.chunker.chunk_documents(documents)
        
        for chunk in chunks:
            chunk.metadata = self.clean_metadata(chunk.metadata)
            chunk.metadata["source"] = os.path.basename(self.pdf_path)
            chunk.page_content = self.preserve_text(chunk.page_content)
        
        logger.info(f"Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        return chunks
    
    def setup_qdrant(self, location: str = None, collection: str = None):
        """Setup Qdrant collection"""
        location = location or settings.QDRANT_LOCATION
        collection = collection or settings.COLLECTION_NAME
        
        logger.info(f"Setting up Qdrant at {location}")
        
        self.client = QdrantClient(path=location)
        
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
            logger.info(f"Removed existing collection: {collection}")
        
        vectors_config = VectorParams(
            size=settings.VECTOR_SIZE,
            distance=Distance.COSINE
        )
        
        self.client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config
        )
        
        logger.info(f"Collection '{collection}' created")
    
    def create_embeddings(self):
        """Create HuggingFace embeddings - CPU-ONLY"""
        logger.info("Creating embeddings (CPU)...")
        start_time = time.time()
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'device': 'cpu',
                    'normalize_embeddings': True,
                    'batch_size': settings.BATCH_SIZE
                }
            )
            
            test_embedding = embeddings.embed_query("Test embedding")
            actual_vector_size = len(test_embedding)
            
            if actual_vector_size != settings.VECTOR_SIZE:
                logger.warning(f"Vector size mismatch. Expected {settings.VECTOR_SIZE}, got {actual_vector_size}")
            
            logger.info(f"Embeddings ready in {time.time() - start_time:.2f}s")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def upload_to_qdrant(self, chunks: List[Document], embeddings):
        """Upload documents to Qdrant"""
        logger.info(f"Uploading {len(chunks)} chunks to Qdrant...")
        start_time = time.time()
        
        try:
            self.vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                location=settings.QDRANT_LOCATION,
                collection_name=settings.COLLECTION_NAME,
                distance=Distance.COSINE,
                batch_size=settings.BATCH_SIZE
            )
            logger.info(f"Uploaded {len(chunks)} chunks in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def ingest(self, pdf_path: str, collection_name: str = None, clear_existing: bool = True) -> Optional[any]:
        """Main ingestion pipeline - accepts PDF path as parameter"""
        self.pdf_path = pdf_path
        collection_name = collection_name or settings.COLLECTION_NAME
        
        total_start = time.time()
        
        try:
            # Step 1: Clear existing DB (optional)
            if clear_existing:
                self.clear_existing_db()
            
            # Step 2: Load PDF
            documents = self.load_pdf(pdf_path)
            
            # Step 3: Chunk documents
            chunks = self.chunk_documents(documents)
            
            if not chunks:
                raise ValueError("No chunks created from PDF")
            
            # Step 4: Create embeddings
            embeddings = self.create_embeddings()
            
            # Step 5: Setup Qdrant
            self.setup_qdrant(collection=collection_name)
            
            # Step 6: Upload to Qdrant
            self.upload_to_qdrant(chunks, embeddings)
            
            total_elapsed = time.time() - total_start
            logger.info(f"✅ INGESTION COMPLETE! Time: {total_elapsed:.2f}s, Chunks: {len(chunks)}")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================
# FUNCTION FOR FASTAPI TO CALL
# ============================================

def ingest_document(pdf_path: str, collection_name: str = None) -> dict:
    """
    Function that FastAPI can call to ingest a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        collection_name: Optional collection name (default from settings)
    
    Returns:
        dict: Status and information about the ingestion
    """
    try:
        if not os.path.exists(pdf_path):
            return {
                "status": "error",
                "message": f"PDF not found: {pdf_path}"
            }
        
        ingester = DocumentIngester()
        vector_store = ingester.ingest(
            pdf_path=pdf_path,
            collection_name=collection_name,
            clear_existing=True
        )
        
        if vector_store:
            return {
                "status": "success",
                "message": "Ingestion completed successfully",
                "collection": collection_name or settings.COLLECTION_NAME,
                "pdf": pdf_path
            }
        else:
            return {
                "status": "error",
                "message": "Ingestion failed"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================
# RUN INGESTION (for testing)
# ============================================

if __name__ == "__main__":
    print("\n🔧 Starting ingestion process (CPU-ONLY)...\n")
    
    # Get PDF path from user or environment
    pdf_path = input("Enter PDF path: ").strip()
    
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        exit(1)
    
    result = ingest_document(pdf_path)
    print(f"\nResult: {result}")
