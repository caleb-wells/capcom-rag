"""Measure the pipeline instead of trusting it.

Retrieval eval (no language model needed): for each retrieval mode, how often
a chunk containing the answer lands in the top k, and how high it ranks.

Generation eval (needs a running model): does the system answer what it can,
decline what it cannot, and cite real lines?
"""

from __future__ import annotations

import json
from pathlib import Path

from . import generate
from .retrieve import MODES, Hit, Retriever

QUESTIONS_PATH = Path("eval/questions.json")
DEPTH = 10  # how far down the ranking we look


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def first_relevant_rank(hits: list[Hit], gold: list[str]) -> int | None:
    """1-based rank of the first chunk containing a gold timestamp."""
    wanted = set(gold)
    for rank, hit in enumerate(hits, start=1):
        if wanted & set(hit.chunk.timestamps):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


def evaluate_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    answerable = [q for q in questions if q["gold"]]
    results: dict[str, dict] = {}
    for mode in MODES:
        ranks = {
            q["q"]: first_relevant_rank(retriever.search(q["q"], DEPTH, mode), q["gold"])
            for q in answerable
        }
        results[mode] = {"all": retrieval_metrics(list(ranks.values()))}
        for kind in ("lexical", "paraphrase"):
            subset = [ranks[q["q"]] for q in answerable if q["kind"] == kind]
            results[mode][kind] = retrieval_metrics(subset)
        results[mode]["misses"] = [q for q, rank in ranks.items() if rank is None]
    return results


def evaluate_generation(retriever: Retriever, questions: list[dict], k: int = 5) -> dict:
    rows = []
    for q in questions:
        hits = retriever.search(q["q"], k)
        result = generate.answer(q["q"], hits)
        if result.source == "retrieval-only":
            raise generate.LLMUnavailable(f"no language model reachable at {generate.BASE_URL}")
        rows.append(
            {
                "q": q["q"],
                "answerable": bool(q["gold"]),
                "refused": result.refused,
                "citations": result.citations,
                "invalid_citations": result.invalid_citations,
                "cites_gold": bool(set(result.citations) & set(q["gold"])),
                "answer": result.text,
            }
        )

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable if not r["refused"]]
    cited = sum(len(r["citations"]) + len(r["invalid_citations"]) for r in rows)
    invalid = sum(len(r["invalid_citations"]) for r in rows)
    return {
        "answerable": len(answerable),
        "answered": len(answered),
        "answered_citing_gold": sum(r["cites_gold"] for r in answered),
        "answered_without_citation": sum(not r["citations"] for r in answered),
        "unanswerable": len(unanswerable),
        "correctly_refused": sum(r["refused"] for r in unanswerable),
        "citations": cited,
        "invalid_citations": invalid,
        "rows": rows,
    }


def format_retrieval_table(results: dict) -> str:
    lines = [
        "| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR | Recall@5 lexical | Recall@5 paraphrase |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, r in results.items():
        a = r["all"]
        lines.append(
            f"| {mode} | {a['recall@1']:.2f} | {a['recall@5']:.2f} | {a['recall@10']:.2f} "
            f"| {a['mrr']:.2f} | {r['lexical']['recall@5']:.2f} | {r['paraphrase']['recall@5']:.2f} |"
        )
    return "\n".join(lines)
