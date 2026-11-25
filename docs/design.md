# Design: capcom-rag

## Intent

A small retrieval-augmented generation system, built by hand, that answers
questions about the Apollo 11 mission from the air-to-ground voice transcript
and cites the exact moments it relied on. It is a public example of how RAG
works and of the judgement calls that make it work well. A reviewer should be
able to read the whole thing in about ten minutes and run it on a fresh clone.

## Goals

- Every stage is visible in plain Python: parse, chunk, index, retrieve, fuse,
  rerank, generate. No orchestration framework.
- Answers cite mission timestamps, and every citation is checked against the
  retrieved context before it is shown.
- The system declines to answer when the transcript does not support one.
- Retrieval quality is measured, and the README reports real numbers with an
  ablation across retrieval strategies.
- Everything runs locally. No API key and no paid service.

## Non-goals

- A web UI, a server, or a vector database. The corpus is about 8,500
  utterances; a NumPy matrix is the right size of tool.
- Multi-document ingestion pipelines, PDFs, OCR.
- Fine-tuning or training anything.

## Anti-goals

- Framework glue that hides the mechanics.
- Numbers in the README that were not produced by the code in the repo.
- A demo that only works on the author's machine.

## Approaches considered

1. **Framework (LangChain or LlamaIndex) plus a vector database.** Fast to
   stand up, but the interesting decisions are hidden inside library calls,
   and a vector database is far more machinery than 1,500 chunks need.
2. **Dense retrieval only, hand-built.** Simple and readable, but weak on this
   corpus: questions often hinge on exact tokens ("1202", "Tranquility Base",
   "P64") that embeddings blur and keyword search nails.
3. **Hand-built hybrid retrieval with an eval harness. (Chosen.)** BM25 and
   dense embeddings, fused with reciprocal rank fusion, then an optional
   cross-encoder rerank. Slightly more code, but each piece is short, and the
   ablation table shows what each piece buys.

## Decisions

| Area | Choice | Why |
|---|---|---|
| Corpus | Apollo 11 technical air-to-ground transcript (NASA, public domain), fetched by script from the Spacelog project at a pinned commit | Real, messy, time-stamped dialogue. Not committed to the repo; the script records the source and checksum. |
| Chunking | Sliding window over utterances, split early at long silences, with overlap. Each chunk keeps its start and end timestamp and speakers | Single utterances ("Roger.") carry no meaning alone. Time gaps are natural topic boundaries. Timestamps become the citation key. |
| Keyword search | BM25, implemented in the repo (about 40 lines) | Shows the mechanics; no dependency. |
| Dense search | `fastembed` running a small BGE model in-process on ONNX; brute-force cosine in NumPy | No server needed, so indexing, tests and retrieval eval work on any machine. |
| Fusion | Reciprocal rank fusion | Needs no score calibration between BM25 and cosine. |
| Rerank | Optional cross-encoder via `fastembed` | Measured in the ablation rather than assumed helpful. |
| Generation | Any OpenAI-compatible endpoint, defaulting to LM Studio on `localhost:1234` with an open-weight model. Called with the standard library, no SDK | Works unchanged with LM Studio, Ollama or llama.cpp. |
| No model running | `ask` falls back to showing the retrieved passages, and says so | The demo never hard-fails on a fresh clone. |
| Grounding | Prompt requires `[DD:HH:MM:SS]` citations and an explicit "not in the transcript" answer. Code then verifies each cited timestamp falls inside a retrieved chunk | Prompting alone is not a guarantee; the check is. |
| Refusal | Two layers: a retrieval-score floor, then the model's own refusal | Cheap guard first, model judgement second. |
| Interface | CLI: `fetch`, `index`, `ask`, `eval` | Smallest thing to review. |
| Eval | Hand-written questions with gold timestamp ranges verified against the transcript, plus unanswerable questions. Reports recall@k and MRR for BM25, dense, hybrid, and hybrid plus rerank; and, when a model is running, citation validity and refusal accuracy | The ablation is the evidence that the design choices matter. |
| Tooling | Python 3.11+, `venv` and `pip`, `pytest` | Lowest-friction setup. |

## Layout

```
capcom_rag/
  corpus.py     parse transcript into utterances
  chunking.py   utterances -> chunks
  bm25.py       keyword index
  embed.py      dense embeddings and rerank (fastembed)
  retrieve.py   hybrid search, RRF, rerank
  generate.py   prompt, LLM call, citation check, fallback
  cli.py        fetch / index / ask / eval
eval/
  questions.json
  run_eval.py
tests/
docs/design.md
README.md
```

## Left to the builder

Chunk size and overlap, the silence threshold, `k` values, the RRF constant
and the refusal floor. Pick sensible starting values, then tune against the
eval set and record what was tried.
