import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

from evaluation.test import load_tests
from implementation.answer import answer_question  # ✅ uses your actual app pipeline


REPORT_PATH = Path(__file__).parent / "eval_report.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def cosine_sim(a: List[float], b: List[float]) -> float:
    return float(dot(a, b) / (norm(a) * norm(b) + 1e-12))


def keyword_hits_in_context(keywords: List[str], context_docs: List[Any]) -> Dict[str, bool]:
    # context_docs are Result objects with .page_content
    joined = "\n".join(getattr(d, "page_content", "") for d in context_docs)
    joined_n = normalize_text(joined)
    hits = {}
    for kw in keywords:
        hits[kw] = normalize_text(kw) in joined_n
    return hits


@dataclass
class EvalRow:
    question: str
    category: str
    reference_answer: str
    predicted_answer: str
    keyword_hits: Dict[str, bool]
    keyword_recall: float
    answer_ref_cosine: float
    passed_retrieval: bool


def run():
    tests = load_tests()
    rows: List[EvalRow] = []

    for t in tests:
        # no history for eval
        pred, ctx = answer_question(t.question, history=[])

        hits = keyword_hits_in_context(t.keywords, ctx)
        recall = sum(hits.values()) / max(len(hits), 1)

        # embedding similarity between predicted answer and reference
        a_vec = embedder.encode(pred, normalize_embeddings=True)
        r_vec = embedder.encode(t.reference_answer, normalize_embeddings=True)
        sim = cosine_sim(a_vec, r_vec)

        row = EvalRow(
            question=t.question,
            category=t.category,
            reference_answer=t.reference_answer,
            predicted_answer=pred,
            keyword_hits=hits,
            keyword_recall=recall,
            answer_ref_cosine=sim,
            passed_retrieval=(recall >= 0.8),  # tweak threshold if needed
        )
        rows.append(row)

        print(f"[{t.category}] recall={recall:.2f} sim={sim:.3f} | {t.question}")

    # aggregate
    avg_recall = sum(r.keyword_recall for r in rows) / max(len(rows), 1)
    avg_sim = sum(r.answer_ref_cosine for r in rows) / max(len(rows), 1)
    retrieval_pass_rate = sum(1 for r in rows if r.passed_retrieval) / max(len(rows), 1)

    report = {
        "summary": {
            "num_tests": len(rows),
            "avg_keyword_recall": avg_recall,
            "avg_answer_ref_cosine": avg_sim,
            "retrieval_pass_rate": retrieval_pass_rate,
        },
        "rows": [asdict(r) for r in rows],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    run()
