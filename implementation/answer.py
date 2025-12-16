from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

# =========================
# Configuration
# =========================

MODEL = "groq/openai/gpt-oss-120b"

DB_NAME = str(Path(__file__).parent.parent / "vector_db_v2")
COLLECTION_NAME = "docs"

RETRIEVAL_K = 20
FINAL_K = 10

WAIT = wait_exponential(multiplier=1, min=10, max=240)

# =========================
# Embeddings (FREE, consistent with ingest)
# =========================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Vector DB
# =========================

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

# =========================
# Prompts
# =========================

SYSTEM_PROMPT = """
You are a knowledgeable assistant representing the company Insurellm.
You must answer the user's question using ONLY the provided context.
Your answer will be evaluated for accuracy, relevance, and completeness.

If the context does not contain the answer, say you do not know.

Context:
{context}
"""

RERANK_SYSTEM_PROMPT = """
You are a document re-ranker.
You are given a question and multiple text chunks.
Rank all chunks by relevance to the question, from most relevant to least relevant.
Reply ONLY with the ordered list of chunk IDs.
"""

# =========================
# Models
# =========================

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk IDs ranked from most relevant to least relevant"
    )

# =========================
# Query Rewriting
# =========================

@retry(wait=WAIT)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    history = history or []

    prompt = f"""
You are answering questions about the company Insurellm.

Conversation history:
{history}

User question:
{question}

Rewrite the question into a short, precise knowledge-base search query.
Respond ONLY with the rewritten query.
"""

    response = completion(
        model=MODEL,
        messages=[{"role": "system", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

# =========================
# Retrieval
# =========================

def fetch_context_unranked(query_text: str) -> list[Result]:
    query_embedding = embedder.encode(
        query_text, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K,
    )

    chunks: list[Result] = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))

    return chunks


def merge_chunks(primary: list[Result], secondary: list[Result]) -> list[Result]:
    seen = set()
    merged = []

    for chunk in primary + secondary:
        key = chunk.page_content
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    return merged

# =========================
# Reranking
# =========================

@retry(wait=WAIT)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    user_prompt = f"Question:\n{question}\n\nChunks:\n"

    for idx, chunk in enumerate(chunks, start=1):
        user_prompt += f"\n# CHUNK {idx}\n{chunk.page_content}\n"

    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=MODEL,
        messages=messages,
        response_format=RankOrder,
    )

    order = RankOrder.model_validate_json(
        response.choices[0].message.content
    ).order

    return [chunks[i - 1] for i in order if 0 < i <= len(chunks)]

# =========================
# RAG Assembly
# =========================

def fetch_context(question: str, history: list[dict]) -> list[Result]:
    rewritten = rewrite_query(question, history)

    original_chunks = fetch_context_unranked(question)
    rewritten_chunks = fetch_context_unranked(rewritten)

    merged = merge_chunks(original_chunks, rewritten_chunks)
    reranked = rerank(question, merged)

    return reranked[:FINAL_K]


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]):
    context = "\n\n".join(
        f"Source: {c.metadata.get('source', 'unknown')}\n{c.page_content}"
        for c in chunks
    )

    system_prompt = SYSTEM_PROMPT.format(context=context)

    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

# =========================
# Public API
# =========================

@retry(wait=WAIT)
def answer_question(question: str, history: list[dict] | None = None):
    history = history or []

    chunks = fetch_context(question, history)
    messages = make_rag_messages(question, history, chunks)

    response = completion(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content, chunks
