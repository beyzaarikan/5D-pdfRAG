# 🏢 5D Institute – AI Knowledge Base

A local Retrieval-Augmented Generation (RAG) system that lets you ask questions about your company's PDF documents. Runs entirely on your machine — no data leaves your environment.

---

## How It Works

```
PDFs → PyMuPDF → Chunk → nomic-embed-text → ChromaDB
                                                  ↓
User question → Embed → Search ChromaDB → Mistral → Answer
```

1. PDFs are loaded, split into chunks, and stored in a local ChromaDB vector database.
2. When a user asks a question, the most relevant chunks are retrieved.
3. The retrieved context is passed to a local Mistral LLM via Ollama to generate an answer.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

---

## Installation

**1. Clone the repository**
```bash
git clone <repo-url>
cd rib_ai_project
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Pull the required Ollama models**
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**4. Create a `.env` file** (optional — defaults will be used if not provided)
```env
PERSIST_DIRECTORY=./db
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RAW_DATA_PATH=./data/raw
```

---

## Usage

**1. Start Ollama**
```bash
ollama serve
```

**2. (Optional) Pre-load PDFs manually**

Place your PDF files in `data/raw/` and run:
```bash
python3 -m src.ingestion.db_creator
```

**3. Start the app**
```bash
streamlit run app.py
```

**4. Upload PDFs via the UI**

You can also upload PDFs directly from the sidebar in the app — they will be automatically processed and added to the database.

---
<img width="948" height="888" alt="Ekran görüntüsü 2026-05-10 195750" src="https://github.com/user-attachments/assets/90d78b72-cf9a-4d51-a897-4e56c6bab448" />




## Project Structure

```
rib_ai_project/
├── app.py                        # Streamlit UI
├── data/
│   └── raw/                      # PDF storage
├── db/                           # ChromaDB vector database (auto-created)
├── src/
│   ├── config.py                 # Configuration via .env
│   ├── ingestion/
│   │   ├── data_loader.py        # PDF loading and chunking
│   │   └── db_creator.py        # Vector database creation
│   └── retrieval/
│       ├── rag_engine.py         # LLM chain and prompt logic
│       └── search.py             # Standalone similarity search
├── .env                          # Environment variables (not committed)
├── requirements.txt
└── README.md
```

---

## Features

- 🔒 Fully local — no API keys, no data sent externally
- 🌐 German and English interface + prompts
- 📄 PDF upload directly from the UI
- 📍 Source citations (filename + page number) for every answer
- ⚡ Cached model loading — fast after first startup
- 🔁 Duplicate-safe ingestion — same PDF can be uploaded without creating duplicate entries

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Ollama + Mistral |
| Embeddings | nomic-embed-text |
| Vector DB | ChromaDB |
| PDF Loader | PyMuPDF |
| LLM Framework | LangChain |
