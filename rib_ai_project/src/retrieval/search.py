from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import Config

def simple_search(query):
    persist_directory = Config.PERSIST_DIR
    
    # 1. Aynı embedding modelini tanımla
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    
    # 2. Var olan veritabanına bağlan
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    if vector_db._collection.count() == 0:
        print("DB is empty, first run data_loader.py.")
        return 
    
    # 3. Arama yap (En yakın 3 parçayı getir)
    print(f"\nQuestion: {query}")
    print("-" * 30)
    
    results = vector_db.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} (Score: {score:.3f}):")
        print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
        print(f"Content: {doc.page_content[:200]}...")



if __name__ == "__main__":
    # Veritabanına PDF'i yükledikten sonra burayı çalıştır
    user_query = """Was ist ein von Bosch entwickeltes           dezentrales, 
nachrichtenbasiertes Bussystem für die Automobilindustrie
und findet wegen seiner Zuverlässigkeit auch in anderen Industrien
weitreichende Verwendung""" # PDF'inde olan bir şeyi sor
    simple_search(user_query)