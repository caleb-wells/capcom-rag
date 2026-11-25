"""Experiment: does searching with a model-drafted excerpt find more answers?

The hard questions in the eval set are paraphrases: "What did Armstrong report
the moment the lunar module was down?" shares no distinctive word with "The Eagle has
landed." This is the HyDE idea (hypothetical document embeddings): ask a
language model to draft what the answer might look like, then search with the
draft as well as the question.

Run from the repo root, after `python -m capcom_rag index`:

    python experiments/query_expansion.py

The drafts are cached in query_expansion_drafts.json, so the numbers reproduce
without a language model. Delete the file to regenerate them.

This is kept as an experiment, not a feature. See the README for why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capcom_rag import embed, evaluate, generate  # noqa: E402
from capcom_rag.retrieve import CANDIDATES, Hit, Retriever, reciprocal_rank_fusion  # noqa: E402

DRAFTS_PATH = Path(__file__).with_name("query_expansion_drafts.json")

DRAFT_PROMPT = (
    "You help search the Apollo 11 air-to-ground voice transcript. Given a question, "
    "write a short plausible excerpt of the radio dialogue (2 to 4 lines, as "
    "'Speaker: words') that would answer it, using the phrasing the astronauts and "
    "Houston would actually have used. Output only the excerpt."
)


def load_drafts(questions: list[dict]) -> dict[str, str]:
    drafts = json.loads(DRAFTS_PATH.read_text()) if DRAFTS_PATH.exists() else {}
    for q in questions:
        if q["q"] not in drafts:
            drafts[q["q"]] = generate.chat(
                [{"role": "system", "content": DRAFT_PROMPT}, {"role": "user", "content": q["q"]}]
            )
            DRAFTS_PATH.write_text(json.dumps(drafts, indent=1))
    return drafts


def search(retriever: Retriever, question: str, draft: str, k: int, rerank: bool) -> list[Hit]:
    """Fuse four rankings: keyword and dense, for the question and for the draft."""
    question_sims = retriever.vectors @ embed.embed_query(question)
    rankings = [
        [doc for doc, _ in retriever.bm25.search(question, CANDIDATES)],
        [int(i) for i in np.argsort(-question_sims)[:CANDIDATES]],
    ]
    if draft:
        draft_sims = retriever.vectors @ embed.embed_documents([draft])[0]
        rankings += [
            [doc for doc, _ in retriever.bm25.search(draft, CANDIDATES)],
            [int(i) for i in np.argsort(-draft_sims)[:CANDIDATES]],
        ]
    ranked = reciprocal_rank_fusion(rankings)
    if rerank:
        shortlist = [doc for doc, _ in ranked[:CANDIDATES]]
        texts = [retriever.chunks[doc].search_text for doc in shortlist]
        ranked = sorted(zip(shortlist, embed.rerank_scores(question, texts)), key=lambda x: -x[1])
    return [Hit(retriever.chunks[doc], score, float(question_sims[doc])) for doc, score in ranked[:k]]


def main() -> None:
    retriever = Retriever.load()
    questions = [q for q in evaluate.load_questions() if q["gold"]]
    drafts = load_drafts(questions)

    print("| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR | Recall@5 lexical | Recall@5 paraphrase |")
    print("|---|---|---|---|---|---|---|")
    for label, rerank in (("hybrid + drafted excerpt", False), ("hybrid + drafted excerpt + rerank", True)):
        ranks = {
            q["q"]: evaluate.first_relevant_rank(
                search(retriever, q["q"], drafts[q["q"]], evaluate.DEPTH, rerank), q["gold"]
            )
            for q in questions
        }
        overall = evaluate.retrieval_metrics(list(ranks.values()))
        by_kind = {
            kind: evaluate.retrieval_metrics([ranks[q["q"]] for q in questions if q["kind"] == kind])
            for kind in ("lexical", "paraphrase")
        }
        print(
            f"| {label} | {overall['recall@1']:.2f} | {overall['recall@5']:.2f} "
            f"| {overall['recall@10']:.2f} | {overall['mrr']:.2f} "
            f"| {by_kind['lexical']['recall@5']:.2f} | {by_kind['paraphrase']['recall@5']:.2f} |"
        )
    print(f"\n{sum(not d for d in drafts.values())} of {len(drafts)} drafts came back empty.")


if __name__ == "__main__":
    main()
