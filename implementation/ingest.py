from __future__ import annotations

import json
import os
from pathlib import Path
from multiprocessing import Pool

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

load_dotenv(override=True)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
DB_NAME = str(Path(__file__).parent.parent / "vector_db_v2")
COLLECTION_NAME = "docs"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 100
WORKERS = int(os.getenv("WORKERS", "5"))  # keep 1 for stable debugging
WAIT = wait_exponential(multiplier=1, min=10, max=240)

llm = ChatOpenAI(
    model_name=MODEL,
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ✅ Free embeddings only
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class Chunk(BaseModel):
    headline: str = Field(description="Brief heading for the chunk")
    summary: str = Field(description="Short summary of the chunk")
    original_text: str = Field(description="Exact original text for this chunk")


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"doc_type": doc_type, "source": file.as_posix(), "text": f.read()}
                )
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You split a document into overlapping chunks for a KnowledgeBase.

Document type: {document["doc_type"]}
Source: {document["source"]}

Requirements:
- Cover the ENTIRE document across chunks (do not omit anything).
- Use overlap (~25% or ~50 words).
- Produce roughly {how_many} chunks (can be more/less if needed).
- For each chunk output:
  - headline
  - summary
  - original_text (verbatim from the document)

Return ONLY valid JSON in this exact format:
{{"chunks":[{{"headline":"...","summary":"...","original_text":"..."}}, ...]}}

DOCUMENT:
{document["text"]}
""".strip()


def _extract_json(text: str) -> str:
    """Extract the first JSON object from model output (robust against extra text)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


@retry(wait=WAIT)
def process_document(document: dict) -> list[dict]:
    prompt = make_prompt(document)
    msg = [SystemMessage(content="You output JSON only."), HumanMessage(content=prompt)]
    resp = llm.invoke(msg).content

    json_str = _extract_json(resp)
    parsed = Chunks.model_validate(json.loads(json_str))

    results = []
    for ch in parsed.chunks:
        page_content = f"{ch.headline}\n\n{ch.summary}\n\n{ch.original_text}"
        metadata = {"source": document["source"], "doc_type": document["doc_type"]}
        results.append({"page_content": page_content, "metadata": metadata})
    return results


def create_chunks(documents: list[dict]) -> list[dict]:
    chunks = []
    if WORKERS <= 1:
        for d in tqdm(documents, total=len(documents)):
            chunks.extend(process_document(d))
        return chunks

    with Pool(processes=WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(res)
    return chunks


def create_embeddings(chunks: list[dict]):
    chroma = PersistentClient(path=DB_NAME)

    # recreate collection
    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        chroma.delete_collection(COLLECTION_NAME)

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    vectors = embedder.encode(texts, normalize_embeddings=True).tolist()

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    print("Ingestion complete")
