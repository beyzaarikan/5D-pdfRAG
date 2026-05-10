import os
from dotenv import load_dotenv

load_dotenv() # .env dosyasındaki verileri okur

class Config:
    PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./db")
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    try:
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
    except ValueError:
        CHUNK_SIZE = 512
        CHUNK_OVERLAP = 64
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "./data/raw")