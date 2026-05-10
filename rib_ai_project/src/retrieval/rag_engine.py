from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import Config

def get_rag_chain():
    persist_directory = Config.PERSIST_DIR
    
    # 1. Hafıza (ChromaDB)
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # 2. Beyin (Mistral)
    llm = ChatOllama(model=Config.MODEL_NAME, temperature=0) # 0 sıcaklık: Uydurmayı azaltır, net cevap verir.

    # 3. Talimat (Prompt)
    template = """
    Sie sind technischer Assistent beim 5D Institute.

    Nutzen Sie die unten stehenden Dokumente, um Benutzerfragen zu beantworten.

    Geben Sie ehrliche, professionelle und prägnante Antworten.Antworten Sie in der Sprache der Frage (Deutsch oder Englisch).

    Wenn Sie die Antwort nicht wissen, schreiben Sie: „Leider konnte ich dazu keine Informationen in der Dokumentation finden.“
    Erfinden Sie keine Informationen, die nicht im CONTEXT stehen.
    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    # 4. Zincir (Chain) - LangChain'in en güçlü olduğu yer
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    # Bu zincir: Soruyu alır -> DB'de arar -> Prompt'a yerleştirir -> LLM'e gönderir
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

if __name__ == "__main__":

    chain, retriever = get_rag_chain()
    
    # Test sorusu
    user_input = "Was ist das von Bosch entwickelte dezentrale, nachrichtenbasierte Bussystem für die Automobilindustrie, das wegen seiner Zuverlässigkeit auch in anderen Industrien weitreichende Verwendung findet?"
    print("\nAI Thinking...")
    source_docs = retriever.invoke(user_input)
    response = chain.invoke(user_input)
   
    for doc in source_docs:
        print(f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
    
    print(f"\nAI Response: \n{response}")