from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, wait_exponential
from pydantic import BaseModel, Field

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(override=True)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
DB_NAME = str(Path(__file__).parent.parent / "vector_db_v2")
COLLECTION_NAME = "docs"

RETRIEVAL_K = 20
FINAL_K = 10

WAIT = wait_exponential(multiplier=1, min=10, max=240)

llm = ChatOpenAI(
    model_name=MODEL,
    temperature=0,
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ✅ free embeddings only
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


SYSTEM_PROMPT = """
You are a knowledgeable assistant representing the company Insurellm.
Answer using ONLY the provided context. If the context doesn't contain the answer, say you don't know.

Context:
{context}
""".strip()


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(description="Chunk IDs ranked most relevant to least relevant")


@retry(wait=WAIT)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    history = history or []
    prompt = f"""
Conversation history:
{history}

User question:
{question}

Rewrite the question into a short, precise knowledge-base search query.
Return ONLY the rewritten query.
""".strip()

    resp = llm.invoke([SystemMessage(content=prompt)]).content
    return resp.strip()


def fetch_context_unranked(query_text: str) -> list[Result]:
    qvec = embedder.encode(query_text, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[qvec], n_results=RETRIEVAL_K)

    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks


def merge_chunks(a: list[Result], b: list[Result]) -> list[Result]:
    seen = set()
    merged = []
    for c in a + b:
        if c.page_content not in seen:
            seen.add(c.page_content)
            merged.append(c)
    return merged


@retry(wait=WAIT)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    sys = """
You are a document re-ranker.
Given a question and chunks with IDs, return ONLY JSON:
{"order":[id1,id2,...]} containing ALL ids ranked most relevant to least.
""".strip()

    user = f"Question:\n{question}\n\nChunks:\n"
    for i, c in enumerate(chunks, start=1):
        user += f"\n# CHUNK ID {i}\n{c.page_content}\n"

    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)]).content.strip()

    # very simple JSON extraction
    start = resp.find("{")
    end = resp.rfind("}")
    if start == -1 or end == -1:
        return chunks  # fallback: keep original order

    import json
    try:
        order = RankOrder.model_validate(json.loads(resp[start:end+1])).order
        return [chunks[i - 1] for i in order if 0 < i <= len(chunks)]
    except Exception:
        return chunks


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]):
    context = "\n\n".join(
        f"Source: {c.metadata.get('source', 'unknown')}\n{c.page_content}"
        for c in chunks
    )
    sys = SYSTEM_PROMPT.format(context=context)

    msgs = [SystemMessage(content=sys)]
    # history is dicts like {"role":"user","content":"..."} already from streamlit
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        else:
            # treat any non-user as assistant context
            from langchain_core.messages import AIMessage
            msgs.append(AIMessage(content=content))

    msgs.append(HumanMessage(content=question))
    return msgs


def fetch_context(question: str, history: list[dict]) -> list[Result]:
    rewritten = rewrite_query(question, history)
    c1 = fetch_context_unranked(question)
    c2 = fetch_context_unranked(rewritten)
    merged = merge_chunks(c1, c2)
    reranked = rerank(question, merged)
    return reranked[:FINAL_K]


@retry(wait=WAIT)
def answer_question(question: str, history: list[dict] | None = None):
    history = history or []
    chunks = fetch_context(question, history)
    msgs = make_rag_messages(question, history, chunks)
    resp = llm.invoke(msgs).content
    return resp, chunks
