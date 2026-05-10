import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.ingestion.data_loader import load_and_split_pdf
from src.config import Config
import glob
import hashlib

def make_chunk_id(pdf_path, chunk_index, content):
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{os.path.basename(pdf_path)}_chunk_{chunk_index}_{hash_val}"

def create_vector_db():
    # 1. Ayarlar
    raw_pdf_path = Config.RAW_DATA_PATH
    persist_directory = Config.PERSIST_DIR # Veritabanının kaydedileceği klasör
    
    pdf_files = glob.glob(os.path.join(raw_pdf_path, "*.pdf"))
    if not pdf_files:
        print(f"Error: '{raw_pdf_path}' could not be found or is empty.")
        return None
    
    print(f"{len(pdf_files)} PDF bulundu: {[os.path.basename(f) for f in pdf_files]}")


    # 2. PDF'i yükle ve parçala (Senin yazdığın fonksiyon)
    all_chunks = []
    all_ids = []

    for pdf_path in pdf_files:
        chunks = load_and_split_pdf(pdf_path)
        ids = [make_chunk_id(pdf_path, i, chunks[i].page_content) for i in range(len(chunks))]
        all_chunks.extend(chunks)
        all_ids.extend(ids)

    print(f"\nall chunks count: {len(all_chunks)}")

    # 3. Embedding modelini tanımla (Ollama açık olmalı!)
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    print("Data is being vectorized and saved to the database... This may take a while.")
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Existing DB found, adding new documents...")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vector_db.add_documents(documents=all_chunks, ids=all_ids)
    else:
        print("Creating new DB...")
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            ids=all_ids,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
    print(f"Success! Database saved to '{persist_directory}' directory.")
    return vector_db

if __name__ == "__main__":
    create_vector_db()