from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Qdrant settings
    QDRANT_LOCATION: str = "./qdrant_data"
    COLLECTION_NAME: str = "labour_code"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    VECTOR_SIZE: int = 1024
    
    # Chunking settings
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200
    
    # CPU settings
    BATCH_SIZE: int = 32
    
    # API settings
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
