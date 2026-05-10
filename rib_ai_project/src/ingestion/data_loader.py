import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config
from langdetect import detect
import logging
logger = logging.getLogger(__name__) 


def load_and_split_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF bulunamadı: {file_path}")

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError(f"PDF okunamadı veya boş: {file_path}")
    
    # 2. Parçalama ayarları
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP ,
        add_start_index=True,
        length_function=len
    )
    
    # ÖNCE parçala, SONRA üzerinde gez
    chunks = text_splitter.split_documents(docs)
    
    # 3. Metadata ekleme (Döngü burada olmalı)
    for chunk in chunks:
        chunk.metadata["type"] = "pdf"
        try:
            chunk.metadata["language"] = detect(chunk.page_content)
        except Exception as e:
            logger.warning(f"Dil tespiti başarısız: {e}")
            chunk.metadata["language"] = "de"
    
    print(f"Biten dosya: {file_path} - Toplam {len(chunks)} parça oluşturuldu.")
    return chunks # Tüm parçaları topluca en son döndür


# Test etmek için:
if __name__ == "__main__":
    # data/raw klasörüne bir PDF koy ve ismini buraya yaz
    test_pdf = "data/raw/V8.pdf" 
    if os.path.exists(test_pdf):
        result = load_and_split_pdf(test_pdf)
        # İlk parçanın içeriğini görelim
        print(f"İlk parça örneği: {result[0].page_content[:100]}...")
        print(f"Metadata örneği: {result[0].metadata}")