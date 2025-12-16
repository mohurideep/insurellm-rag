from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

# LLM used ONLY for chunking (semantic splitting + headline + summary)
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

DB_NAME = str(Path(__file__).parent.parent / "vector_db_v2")  # keep same if you want
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

AVERAGE_CHUNK_SIZE = 100
WORKERS = int(os.getenv("WORKERS", "3"))
WAIT = wait_exponential(multiplier=1, min=10, max=240)

# ✅ FREE embeddings only (MiniLM) — must match answer.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_document(self, source: str, doc_type: str) -> Document:
        content = f"{self.headline}\n\n{self.summary}\n\n{self.original_text}"
        meta = {"source": source, "doc_type": doc_type}
        return Document(page_content=content, metadata=meta)


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
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
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["doc_type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with JSON only that matches this schema:
{{"chunks":[{{"headline":"...", "summary":"...", "original_text":"..."}}, ...]}}
""".strip()


def make_messages(document: dict):
    return [{"role": "user", "content": make_prompt(document)}]


@retry(wait=WAIT)
def process_document(document: dict) -> list[Document]:
    response = completion(model=MODEL, messages=make_messages(document), response_format=Chunks)
    reply = response.choices[0].message.content

    parsed = Chunks.model_validate_json(reply).chunks
    return [c.as_document(source=document["source"], doc_type=document["doc_type"]) for c in parsed]


def create_chunks(documents: list[dict]) -> list[Document]:
    """
    Create semantic chunks using a number of workers in parallel.
    If you get rate limits or pickling issues, set WORKERS=1.
    """
    chunks: list[Document] = []
    if WORKERS <= 1:
        for d in tqdm(documents, total=len(documents)):
            chunks.extend(process_document(d))
        return chunks

    with Pool(processes=WORKERS) as pool:
        for docs in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(docs)
    return chunks


def create_embeddings(chunks: list[Document]):
    # wipe old db collection by deleting whole folder (simplest + reliable)
    # If you want to keep v1 and v2 DB separate, change DB_NAME to vector_db_v2.
    if os.path.exists(DB_NAME):
        # safer: delete collection via Chroma API by recreating, but folder delete is reliable
        import shutil
        shutil.rmtree(DB_NAME, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    collection = vectorstore._collection
    print(f"Vectorstore created with {collection.count()} documents")
    return vectorstore


if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    print("Ingestion complete")
