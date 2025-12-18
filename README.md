# Insurellm-RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) system for the Insurellm knowledge base.  
This project ingests structured content, embeds it with open-source embeddings, and generates grounded answers using an OpenAI-compatible LLM backend.

Versioning:
- **v1.0:** Basic RAG with fixed chunking
- **v2.0:** Query rewriting and LLM-based reranking
- **v2.2:** Streamlit evaluation dashboard + metrics engine

---

## 📌 Table of Contents

- [Features](#features)  
- [Architecture](#architecture)  
- [Setup & Run](#setup--run)  
  - Requirements  
  - Environment  
  - Ingestion  
  - Chat UI  
  - Evaluation UI  
- [Evaluation](#evaluation)  
- [Versioning & Releases](#versioning--releases)  
- [Directory Layout](#directory-layout)  
- [Troubleshooting](#troubleshooting)

---

## 🚀 Features

- **Semantic chunking** of documents using an LLM
- **Free embeddings** via HuggingFace `all-MiniLM-L6-v2`
- **Dual retrieval:** original + rewritten query
- **LLM reranking** of retrieved chunks
- **Streamlit chat interface**
- **Evaluation dashboard** with progress, charts, and metrics

---

## 🧱 Architecture

   +----------------------+
   | Knowledge Base (md)  |
   +-----------+----------+
               |
        Ingestion (v2) — semantic chunks
               |
 embeddings → Chroma persistent vector db
               |
 Chat UI        |     Evaluation UI
   |            |         |
answer_question | evaluate_iter()
| | |
rewrite → retrieve → rerank → answer


- **Embeddings:** open-source `all-MiniLM-L6-v2`
- **LLM:** OpenAI-compatible (`openai/gpt-oss-120b`)
- **Vector store:** Chroma

---

## 🛠️ Setup & Run

### Requirements

Python 3.10+, recommended venv:

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.\.venv\Scripts\activate      # Windows
