import streamlit as st
import os
from langchain_chroma import Chroma
from src.config import Config
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Hataları dosyaya yazar
        logging.StreamHandler()        # Terminale basar
    ]
)

logger = logging.getLogger(__name__)

# --- AYARLAR ---
PERSIST_DIR = Config.PERSIST_DIR
MODEL = Config.MODEL_NAME

DE_PROMPT = """Sie sind technischer Assistent beim 5D Institute.
Nutzen Sie die unten stehenden Dokumente, um Benutzerfragen zu beantworten.
Erfinden Sie keine Informationen. Wenn die Antwort nicht im Kontext steht, sagen Sie es ehrlich.

CONTEXT: {context}
FRAGE: {question}
ANTWORT:"""

EN_PROMPT = """You are a technical assistant at 5D Institute.
Use the documents below to answer user questions.
Do not invent information. If it's not in the context, say you don't know.

CONTEXT: {context}
QUESTION: {question}
ANSWER:"""

@st.cache_resource(show_spinner=False)
def load_chain(language: str):
    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    if vector_db._collection.count() == 0:
        raise ValueError("database is empty! Please run vector_db.py first.")
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=Config.MODEL_NAME, temperature=0, streaming=True)
    template = DE_PROMPT if language == "Deutsch" else EN_PROMPT
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm_chain = prompt | llm | StrOutputParser()
    return retriever, llm_chain, format_docs


# --- UI ---
st.set_page_config(page_title="5D AI Assistant", layout="wide", page_icon="🤖")

st.markdown("""<style> .stChatFloatingInputContainer {padding-bottom: 20px;} </style>""", unsafe_allow_html=True)

st.title("🏢 5D Institute – AI Knowledge Base")

# Sidebar
language = st.sidebar.selectbox("🌐 Sprache / Language", ["Deutsch", "English"])

if st.sidebar.button("🗑️ Clear Chat / Verlauf löschen"):
    st.session_state.messages = []
    st.rerun()

if language != st.session_state.get("last_language"):
    st.session_state.messages = []
    st.session_state["last_language"] = language
    load_chain.clear()  # ← BU SATIRI EKLE
    st.rerun()          

st.sidebar.markdown("---")
st.sidebar.caption(f"🧠 Engine: {Config.MODEL_NAME} | Local RAG System")

retriever, llm_chain, format_docs = load_chain(language)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesaj Geçmişi
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            expander_title = "📄 Quellen" if language == "Deutsch" else "📄 Sources"
            with st.expander(expander_title):
                for s in msg["sources"]:
                    st.caption(f"📍 {s['source']} — Page {s['page']}")

# Chat Girişi
if user_input := st.chat_input("Frage stellen..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer=""
        sources = []
        try:
            source_docs = retriever.invoke(user_input)
            context = format_docs(source_docs)

            with st.spinner("Denken..." if language == "Deutsch" else "Thinking..."):
                source_docs = retriever.invoke(user_input)
                context = format_docs(source_docs)

            answer = st.write_stream(
                llm_chain.stream({"context": context, "question": user_input})
            )

            sources = [
                {"source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", "?")}
                for doc in source_docs
            ]

            expander_title = "📄 Quellen" if language == "Deutsch" else "📄 Sources"
            with st.expander(expander_title):
                for s in sources:
                    st.caption(f"📍 {s['source']} — Page {s['page']}")
        except Exception as e:
            logger.error(f"Inference error: {e}")
            answer = "⚠️ Ein Fehler ist aufgetreten." if language == "Deutsch" else "⚠️ An error occurred."
            st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })