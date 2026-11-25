# capcom-rag

Ask questions about the Apollo 11 mission and get answers grounded in the
actual air-to-ground radio transcript, with a citation for every claim.
Everything runs on your own machine using open-weight models.

```
$ python -m capcom_rag ask "What is written on the plaque on the landing gear?"

According to the transcript, the plaque reads: "Here Man from the planet Earth first
set foot upon the Moon, July 1969 A.D. We came in peace for all mankind" [04:13:52:40].
It also features "the crew members' signatures and the signature of the President of
the United States" [04:13:52:40].

Citations verified against retrieved passages: 1
```

```
$ python -m capcom_rag ask "What did the crew say moon dust smelled like inside the cabin?"

The transcript does not answer this.
```

In mission control, CAPCOM (the capsule communicator) was the one person who
spoke to the crew, relaying what the rest of the room knew. This project does
the same job in reverse: it relays what the transcript says, and nothing else.

## What this is

A small retrieval-augmented generation (RAG) system in about 800 lines of
Python, built without a RAG framework, as a worked example of how RAG works and of the
judgement calls that decide whether it works well.

It aims to do three things:

1. **Show every stage in plain code.** There is no LangChain, no LlamaIndex
   and no vector database. Parsing, chunking, keyword search, vector search,
   rank fusion, reranking, prompting and citation checking are each a short
   file you can read top to bottom.
2. **Give answers you can check.** Every answer cites mission timestamps, and
   the code verifies each citation against the passages the model was shown.
   When the transcript does not contain the answer, the system says so.
3. **Measure itself.** A purpose-built evaluation set produces the numbers in
   this README, including the ones that are not flattering.

## How it works

```
                        INDEXING (once)
  transcript ──> parse ──> chunk ──> enrich ──┬──> BM25 keyword index
  8,482 lines             1,327 chunks        └──> embeddings (BGE-small)

                        ANSWERING (per question)
                 ┌──> BM25 top 30 ────┐
  question ──────┤                    ├──> rank fusion ──> rerank ──> top 5
                 └──> vector top 30 ──┘      (RRF)      (cross-encoder)
                                                                        │
  cited answer <── verify citations <── local LLM (Qwen) <── prompt <───┘
```

| Stage | File | What it does, and why |
|---|---|---|
| Parse | `capcom_rag/corpus.py` | Downloads the transcript at a pinned commit, verifies its checksum, and turns it into utterances: timestamp, speaker, text. |
| Chunk | `capcom_rag/chunking.py` | Groups utterances into roughly 160-word stretches of conversation. A lone "Roger." means nothing, so the unit of retrieval is an exchange. A silence of five minutes ends a chunk early, because radio silence is the transcript's natural topic break. Neighbouring chunks share two utterances so an exchange cut by a boundary survives intact in one of them. |
| Enrich | `capcom_rag/chunking.py` | The transcript says `CDR (EVA)`. People ask about "Armstrong" and "the moonwalk". The searchable copy of each chunk swaps in names and opens with one line of context. The copy shown to the reader and the model stays verbatim. |
| Keyword search | `capcom_rag/bm25.py` | BM25, implemented here in about 40 lines. It is the right tool for exact tokens such as "1202" or "P64". |
| Vector search | `capcom_rag/embed.py` | `BAAI/bge-small-en-v1.5` embeddings, searched by brute-force cosine similarity in NumPy. With 1,327 chunks, a vector database would be machinery for its own sake. |
| Fuse | `capcom_rag/retrieve.py` | Reciprocal rank fusion merges the two rankings using ranks alone, so BM25 scores and cosine similarities never need to be put on a common scale. |
| Rerank | `capcom_rag/retrieve.py` | A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question together with each of the 30 fused candidates and reorders them. It is too slow to run on everything and accurate enough to be worth running on a shortlist. |
| Generate | `capcom_rag/generate.py` | The top 5 chunks go to a local open-weight model in mission order. The prompt allows only the excerpts as a source, requires a timestamp citation per claim, and gives the model an exact sentence to use when the answer is not there. |
| Verify | `capcom_rag/generate.py` | Code extracts every cited timestamp and checks it is a real line in a passage the model was given. A citation that fails is flagged to the reader, and so is an answer with no verifiable citation at all. |

### The models

All three are open-weight and run locally. Nothing is sent to a hosted API.

| Job | Model | Runs in |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | In-process (ONNX, via `fastembed`) |
| Reranking | `Xenova/ms-marco-MiniLM-L-6-v2` | In-process (ONNX, via `fastembed`) |
| Answers | Qwen (`qwen3.6-35b-a3b-mlx` during development) | LM Studio's local server |

The answer step speaks the OpenAI-compatible chat API using only the Python
standard library, so it works unchanged with LM Studio, Ollama, llama.cpp or
vLLM, and with whichever open model you have loaded.

## Quickstart

Developed and tested on Python 3.13 on an Apple M3 Max, with `fastembed` 0.9.1,
`numpy` 2.5.5 and `onnxruntime` 1.32.0.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m capcom_rag fetch     # download the transcript (0.9 MB)
.venv/bin/python -m capcom_rag index     # chunk and embed, about 2 minutes on CPU
.venv/bin/python -m capcom_rag ask "What is written on the plaque on the landing gear?"
```

`ask` needs a local model server for the written answer. With LM Studio:

```bash
lms server start
lms load <an open-weight chat model> --context-length 8192
```

With no server running, `ask` still works: it prints the retrieved passages
and says that no model was reachable. Indexing, the retrieval evaluation and
the tests never need a language model.

| Command | What it does |
|---|---|
| `fetch` | Downloads the transcript and verifies its checksum. |
| `index` | Chunks, embeds and saves the index to `.index/`. |
| `ask "..."` | Answers with citations. `--show-passages` prints what was retrieved; `--mode` picks `bm25`, `dense`, `hybrid` or `hybrid+rerank`; `-k` sets how many passages. |
| `eval` | Runs the evaluation. `--retrieval-only` skips the language model. |

To point at a different server or model, set `LLM_BASE_URL` (default
`http://localhost:1234/v1`) and `LLM_MODEL` (default: the first chat model
the server lists).

```bash
.venv/bin/python -m pytest          # 22 tests, under a second, no downloads
```

## Results

The evaluation set (`eval/questions.json`) has 40 questions, written for this project by Claude
and checked against the transcript:

- 14 **lexical** questions that share distinctive words with the answer
  ("What happened to Thor Heyerdahl's papyrus boat?").
- 18 **paraphrase** questions that do not ("How did mission control react
  right after the landing was confirmed?" for the line "You got a bunch of
  guys about to turn blue. We're breathing again.").
- 8 **unanswerable** questions about things the transcript never mentions.

Each answerable question lists the transcript timestamps that contain its
answer. A retrieved chunk counts as a hit if it includes one of them.

### Retrieval

Recall@k is the share of questions with a correct chunk in the top k. MRR
rewards ranking it higher.

| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR | Recall@5 lexical | Recall@5 paraphrase |
|---|---|---|---|---|---|---|
| bm25 | 0.50 | 0.59 | 0.59 | 0.54 | 1.00 | 0.28 |
| dense | 0.28 | 0.41 | 0.53 | 0.34 | 0.57 | 0.28 |
| hybrid | 0.41 | 0.66 | 0.72 | 0.48 | 0.93 | 0.44 |
| hybrid+rerank | 0.50 | 0.66 | 0.78 | 0.59 | 1.00 | 0.39 |

Reproduce with `python -m capcom_rag eval --retrieval-only`.

What the table says:

- **Hybrid beats either retriever alone.** Fusion lifts recall@5 from 0.59
  (BM25) and 0.41 (dense) to 0.66, and recall@10 to 0.72.
- **Reranking improves the top of the list.** Recall@1 goes from 0.41 to 0.50
  and recall@10 from 0.72 to 0.78. It restores lexical recall@5 to 1.00 but
  costs a little on paraphrases (0.44 to 0.39).
- **Keyword-style questions are solved; paraphrases are not.** This corpus is
  hard on embeddings: terse radio calls, jargon and numbers, with the famous
  lines buried among checklist readouts. A question like "What was said when
  the capsule hit the water?" has no surface or semantic overlap with the
  answer, "Splashdown!", inside a chunk of recovery chatter.

Thirty-two questions is a small set. One question moves a recall figure by
three points, and the settings below were tuned against the same set, with no
held-out split. Treat the figures as directional.

### Answers

All 40 questions, through the full pipeline (hybrid+rerank, top 5 passages,
Qwen `qwen3.6-35b-a3b-mlx` in LM Studio). One run; local generation is not
perfectly repeatable. Every answer is in `eval/results.json`.

| Measure | Result |
|---|---|
| Unanswerable questions correctly declined | 8 of 8 |
| Citations that point to a real line in the retrieved passages | 39 of 39 |
| Answerable questions answered | 25 of 32 |
| ...of which cite a timestamp from the answer key | 21 of 25 |
| Answerable questions declined | 7 of 32 |

Reproduce with `python -m capcom_rag eval` while a model server is running.

Reading all 25 answers in full: 22 are correct. One of those names the right
ship from a passage the answer key did not list, which is a gap in the key,
not in the answer. The other 3 are wrong. A script also checked the quoted
text in the answers against the transcript and found every quotation
verbatim.

- **The model never invented a citation and never answered an unanswerable
  question.** That is the behaviour the prompt and the citation check exist
  to produce.
- **All 7 declined questions are retrieval misses.** The right passage was
  not in the top 5, and the model said so instead of guessing. For this kind
  of system that is the right way to fail.
- **The 3 wrong answers are the instructive ones.** Each quotes a real line,
  cites it correctly, and does not answer the question. Asked how Aldrin
  described the landscape in two words, the model was shown a nearby passage
  without "Magnificent desolation" and offered "powdery surface". The
  citation check passes these, because the quote is genuine. Catching them
  takes better retrieval or a second model judging whether the cited line
  supports the claim.

So in this run the quality of the answers was bounded by retrieval: all 21
questions whose answer was retrieved in the top 5 were answered correctly,
and every decline and every wrong answer traces back to a retrieval miss.

## What I tried, and what it taught me

**A retrieval-score floor for refusing questions. Removed.** The plan was to
decline before calling the model whenever the best passage's similarity was
low. Measured, the best-passage similarity ran from 0.54 to 0.79 for
answerable questions and from 0.62 to 0.76 for unanswerable ones; 28 of the
32 answerable questions scored below the highest unanswerable one. Even
plainly off-topic questions ("How do I bake sourdough bread?") scored 0.49 to
0.56, at or just under the bottom of the answerable range, so a floor low
enough to keep every good question would stop almost nothing. No threshold
separates them, so refusal is
left to the model and the citation check. A question about moon dust is
*near* plenty of transcript; whether the transcript *answers* it is a reading
task, not a distance.

**Speaker enrichment. Kept, with a cost.** Compared with indexing the raw
labels (`CDR (EVA): ...`), the enriched index does better where it counts:

| Index | hybrid Recall@5 | hybrid paraphrase Recall@5 | hybrid+rerank Recall@5 | hybrid+rerank Recall@10 | dense Recall@5 |
|---|---|---|---|---|---|
| raw labels | 0.56 | 0.22 | 0.59 | 0.72 | 0.50 |
| enriched (current) | 0.66 | 0.44 | 0.66 | 0.78 | 0.41 |

The last column is the cost. Dense search on its own got worse, and the
current version did not recover it. My best explanation, which I have not
tested, is that the shared names and context sentence make chunks from the
same phase of the mission look more alike to the embedding model. Keyword search gains more than dense search loses, so the fused result
improves. An earlier version spelled the full description out on every line
("Aldrin, lunar module pilot, aboard the lunar module Eagle: Yes."), which in
chunks of two-word radio calls made boilerplate most of the text; in a
development run it reached reranked recall@10 of 0.69 against 0.78 now. That
version is no longer in the code, so that one figure cannot be reproduced
from this tree.

**Smaller chunks. Worse.** Dropping from 160 to 100 words nearly doubled the
chunk count (1,327 to 2,334) and lowered reranked recall@5 from 0.66 to 0.56
and recall@10 from 0.78 to 0.59.

Both comparisons reproduce by building an index in memory with
`Retriever.build(utterances, enrich=False)` or `max_words=100` and passing it
to `evaluate.evaluate_retrieval`.

**Searching with a model-drafted excerpt (HyDE). Promising, not adopted.**
`experiments/query_expansion.py` asks the model to draft what the answer
might sound like, then searches with the draft as well as the question.

| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR | Recall@5 lexical | Recall@5 paraphrase |
|---|---|---|---|---|---|---|
| hybrid + drafted excerpt | 0.50 | 0.75 | 0.78 | 0.59 | 0.86 | 0.67 |
| hybrid + drafted excerpt + rerank | 0.50 | 0.69 | 0.81 | 0.60 | 1.00 | 0.44 |

Paraphrase recall@5 rises from 0.44 to 0.67. It is left out of the main
pipeline for three reasons. It adds a model call of roughly 30 seconds per
question with a local reasoning model. Much of the gain comes from the model
already knowing Apollo 11's famous lines, which would not carry over to a
private corpus. And reranking against the original question undoes most of
the benefit, which suggests the cross-encoder, not recall, is the next thing
to fix. The drafts are cached in the repo, so the table reproduces without a
language model.

## Limitations

- **Citation checking proves a citation is real, not that it is right.** It
  catches invented and misremembered timestamps. It cannot tell whether the
  cited line supports the claim; that needs a human or a judge model.
- **Paraphrased questions often miss**, as measured above.
- **Long utterances are truncated for embedding.** A few news readouts exceed
  the embedding model's 512-token window. BM25 still sees the full text.
- **One mission, one transcript.** The parser is specific to this format. The
  chunker, retrievers, fusion, generation and evaluation are not.

## Project layout

```
capcom_rag/
  corpus.py      fetch and parse the transcript
  chunking.py    utterances -> enriched chunks
  bm25.py        keyword index
  embed.py       embedding and reranking models
  retrieve.py    index build, hybrid search, fusion, rerank
  generate.py    prompt, LLM call, citation check, fallback
  evaluate.py    retrieval and generation metrics
  cli.py         fetch / index / ask / eval
eval/questions.json              the evaluation set
experiments/query_expansion.py   the HyDE experiment
tests/test_pipeline.py           unit tests
docs/design.md                   the design, written before the code
```

## How this was built

This project was built in a working session with
[Claude Code](https://claude.com/claude-code). I set the direction and made
the decisions; Claude wrote the code and ran the experiments.

My calls: the Apollo 11 transcript as the corpus, an open-weight model running
locally instead of a hosted API, a command line as the only interface, and
the name. Claude proposed building the pipeline from first principles instead of on
a framework, wrote that approach up in [`docs/design.md`](docs/design.md)
before any code existed, and I approved it. That document is unchanged in substance, so it
also records where the build departed from the plan: the refusal floor it
describes was measured and removed, and the enrichment step is not in it at
all because the first evaluation run is what showed the need.

The practice I would point to is the evaluation loop. The retrieval numbers
were poor on the first run, and each change after that was kept or dropped on
what the eval set showed, including two ideas that sounded right and were not.

## Data and credits

The transcript is the *Apollo 11 Technical Air-to-Ground Voice Transcription*,
a NASA work in the public domain. This project downloads the cleaned
plain-text edition maintained by the [Spacelog](https://github.com/Spacelog/Spacelog)
project, pinned to a specific commit. The transcript is not stored in this
repository.
