# capcom-rag

Ask questions about the Apollo 11 mission and get answers drawn from the
actual air-to-ground radio transcript, with timestamp citations that the code
checks. The models are open-weight and run on your own machine.

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

In mission control, CAPCOM (the capsule communicator) was normally the only
person who spoke to the crew, relaying what the rest of the room knew. This project does
the same job in reverse: it relays what the transcript says, and nothing else.

## What this is

A small retrieval-augmented generation (RAG) system in about 800 lines of
Python, built without a RAG framework, as a worked example of how RAG works
and of the judgement calls that decide whether it works well.

It aims to do three things:

1. **Show every stage in plain code.** There is no LangChain, no LlamaIndex
   and no vector database: the embeddings live in a NumPy array and are
   searched by brute force. Parsing, chunking, keyword search, vector search,
   rank fusion, reranking, prompting and citation checking sit in eight short
   files you can read top to bottom.
2. **Give answers you can check.** The model is told to cite a mission
   timestamp for each claim, and the code verifies each citation against the
   passages the model was shown. When the passages do not contain the answer,
   the model is told to say so.
3. **Measure itself.** A purpose-built evaluation set produces the numbers in
   this README, including the ones that are not flattering.

## How it works

```
                        INDEXING (once)
  transcript ──> parse ──> chunk ──> enrich ──┬──> BM25 keyword index
  8,482 utterances        1,327 chunks        └──> embeddings (BGE-small)

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
| Chunk | `capcom_rag/chunking.py` | Groups utterances into stretches of conversation of up to about 160 words. A lone "Roger." means nothing, so the unit of retrieval is an exchange. A silence of five minutes ends a chunk early, because radio silence is the transcript's natural topic break. Neighbouring chunks share two utterances so an exchange cut by a boundary survives intact in one of them. |
| Enrich | `capcom_rag/chunking.py` | The transcript says `CDR (EVA)`. People ask about "Armstrong", a name that appears once in the whole transcript. The searchable copy of each chunk swaps in surnames and opens with one line of context, such as "Armstrong is the commander, walking on the lunar surface". The copy shown to the reader and the model stays verbatim. |
| Keyword search | `capcom_rag/bm25.py` | BM25, implemented here in under 60 lines. It suits exact tokens such as "1202" or "P64". |
| Vector search | `capcom_rag/embed.py`, `capcom_rag/retrieve.py` | `BAAI/bge-small-en-v1.5` embeddings, searched by brute-force cosine similarity in NumPy. With 1,327 chunks there is nothing for a vector database to speed up. |
| Fuse | `capcom_rag/retrieve.py` | Reciprocal rank fusion merges the two rankings using ranks alone, so BM25 scores and cosine similarities never need to be put on a common scale. |
| Rerank | `capcom_rag/retrieve.py` | A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question together with each of the top 30 fused candidates and reorders them. Scoring every chunk this way for every question would be far more work, so it only sees a shortlist. The results table shows what it buys. |
| Generate | `capcom_rag/generate.py` | The top 5 chunks go to a local open-weight model in mission order. The prompt allows only the excerpts as a source, requires a timestamp citation per claim, and gives the model an exact sentence to use when the answer is not there. |
| Verify | `capcom_rag/generate.py` | Code extracts every cited timestamp and checks it is a real line in a passage the model was given. A citation that fails is flagged to the reader, and so is an answer with no verifiable citation at all. |

### The models

All three are open-weight and run locally. Questions and answers are not sent
to any hosted API. The network is used only for one-time downloads: the
transcript from GitHub, and the embedding and reranking models from Hugging
Face on first use.

| Job | Model | Runs in |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | In-process (ONNX, via `fastembed`) |
| Reranking | `Xenova/ms-marco-MiniLM-L-6-v2` | In-process (ONNX, via `fastembed`) |
| Answers | Qwen (`qwen3.6-35b-a3b-mlx` during development) | LM Studio's local server |

The answer step speaks the OpenAI-compatible chat API using only the Python
standard library. It was tested only with LM Studio and this one Qwen model.
Other servers that offer the same API (Ollama, llama.cpp, vLLM) and other
models should work, but have not been tried.

## Quickstart

Developed and tested on Python 3.13 on an Apple M3 Max, with `fastembed` 0.9.1,
`numpy` 2.5.5 and `onnxruntime` 1.32.0.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m capcom_rag fetch     # download the transcript (0.9 MB)
.venv/bin/python -m capcom_rag index     # chunk and embed, about 2 minutes on the M3 Max
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

The evaluation set (`eval/questions.json`) has 40 questions, written for this
project by Claude. Each answer timestamp was checked to exist in the
transcript, and the topics of the unanswerable questions were searched for
and not found.

- 14 **lexical** questions that share distinctive words with the answer
  ("What happened to Thor Heyerdahl's papyrus boat?").
- 18 **paraphrase** questions that do not ("How did mission control react
  right after the landing was confirmed?" for the line "You got a bunch of
  guys about to turn blue. We're breathing again.").
- 8 **unanswerable** questions about things the transcript does not cover.

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
- **Keyword-style questions work; paraphrases mostly do not.** All 14 lexical
  questions have a correct chunk in the top 5, against 7 of 18 paraphrases.
  The text is terse radio calls, jargon and numbers. A question like "What
  was said at the very end, when the capsule hit the water?" shares no
  distinctive word with its answer, "Splashdown!".

Thirty-two questions is a small set. One question moves an overall recall
figure by three points, and the lexical and paraphrase columns by seven and
six. The settings below were tuned against the same set, with no held-out
split. Treat the figures as directional.

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

Claude read all 25 answers against the transcript; no one else has reviewed
them. On that reading, 22 answer the question correctly and 3 do not. One of
the 22 names the right ship from a passage the answer key did not list, which
is a gap in the key. At least one of the 22 (the seismometer question) gets
the answer right and then pads it with quotes from neighbouring lines that
are about other equipment. A one-off script, not in this repository, looked
for each quoted span of four or more words in the transcript, ignoring case
and punctuation, and found all of them.

- **In this run the model did not invent a citation or answer an
  unanswerable question.** That is the behaviour the prompt and the citation
  check exist to produce.
- **All 7 declined questions are retrieval misses.** The right passage was
  not in the top 5, and the model declined instead of guessing.
- **The 3 that miss are the instructive ones.** Each quotes a real line and
  cites it correctly, but the line does not answer the question. Asked how
  Aldrin described the landscape in two words, the model was not shown
  "Magnificent desolation" and offered "powdery surface" from a line four
  minutes later. One of the three goes on to say the excerpts do not cover
  the question. The citation check passes all of these, because the quotes
  are genuine. Catching them would take better retrieval or a second model
  judging whether the cited line supports the claim.

So in this run the answers were limited by retrieval: all 21 questions whose
answer was retrieved in the top 5 were answered correctly on Claude's
reading, and every decline and every miss was a question whose answer was
not retrieved.

## What was tried

**A retrieval-score floor for refusing questions. Removed.** The plan was to
decline before calling the model whenever the best passage's similarity was
low. Measured, the best-passage similarity ran from 0.54 to 0.79 for
answerable questions and from 0.62 to 0.76 for unanswerable ones; 28 of the
32 answerable questions scored below the highest unanswerable one. Three
plainly off-topic test questions ("How do I bake sourdough bread?") scored
0.49, 0.55 and 0.56, straddling the bottom of the answerable range. A floor
set just low enough to keep every answerable question would have stopped one
of those three and none of the eight unanswerable ones. So the floor was
removed, and declining is left to the model. A question about moon dust is
*near* plenty of transcript; whether the transcript *answers* it is a reading
task, not a distance.

**Speaker enrichment. Kept, with a cost.** Compared with indexing the raw
labels (`CDR (EVA): ...`), the enriched index does better on the fused
results and worse on dense search alone:

| Index | hybrid Recall@5 | hybrid paraphrase Recall@5 | hybrid+rerank Recall@5 | hybrid+rerank Recall@10 | dense Recall@5 |
|---|---|---|---|---|---|
| raw labels | 0.56 | 0.22 | 0.59 | 0.72 | 0.50 |
| enriched (current) | 0.66 | 0.44 | 0.66 | 0.78 | 0.41 |

The last column is the cost. Dense search on its own got worse, and the
current version did not recover it. One explanation, not tested, is that the
shared names and context sentence make chunks from the same phase of the
mission look more alike to the embedding model. Keyword search improved at
the top of its ranking (BM25 recall@1 went from 0.38 to 0.50), and the fused
results improved.

An earlier version spelled the full description out on every line ("Aldrin,
lunar module pilot, aboard the lunar module Eagle: Yes."), so in chunks of
short radio calls the labels were longer than the dialogue. In a development
run it reached reranked recall@10 of 0.69 against 0.78 now. That
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
question with this local reasoning model. Several drafts reproduce famous
lines word for word, so some of the gain is the model's prior knowledge of
Apollo 11, which would not carry over to a private corpus; how much was not
measured. And reranking against the original question takes the paraphrase
figure back to 0.44. Three of the 32 drafts came back empty; the likely cause,
not confirmed, is the model spending its whole token budget on reasoning. The
drafts are cached in the
repo, so the table reproduces without a language model.

## Limitations

- **Citation checking proves a citation is real, not that it is right.** It
  catches invented and misremembered timestamps. It cannot tell whether the
  cited line supports the claim; that needs a human or a judge model.
- **Paraphrased questions often miss**, as measured above.
- **Long chunks are truncated for embedding.** 11 of the 1,327 chunks exceed
  the embedding model's 512-token window. BM25 still sees the full text.
- **Queries are embedded without an instruction prefix.** BGE models document
  a short instruction to prepend to search queries. The library call used
  here does not add it, and adding it was not tried.
- **Built for one transcript.** BM25, vector search, fusion and reranking are
  general. The parser, the speaker enrichment, the prompt and the evaluation
  questions are specific to Apollo 11.
- **Tested in one setup:** one machine, one Python version, one model server
  and one answer model.

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

This project was built over two days of working sessions between the
repository owner and [Claude Code](https://claude.com/claude-code). This
section, like the rest of the README, was written by Claude.

- **The owner** set the direction, chose the Apollo 11 transcript as the
  corpus and a command line as the only interface (each from options Claude
  offered), asked for an open-weight model running locally instead of a
  hosted API, chose the name, and approved the design before any code was
  written.
- **Claude** proposed building the pipeline without a framework, wrote the
  design in [`docs/design.md`](docs/design.md), made the implementation
  decisions, wrote all of the code, the tests and the evaluation questions,
  and ran the experiments.

The design document is unchanged in substance, so it also records where the
build departed from the plan: the refusal floor it describes was measured and
removed, and the enrichment step is not in it at all, because it was added
after the first evaluation run.

After that first run, each change was kept or dropped according to what the
evaluation set showed.

## Data and credits

The transcript is the *Apollo 11 Technical Air-to-Ground Voice Transcription*,
a NASA work in the public domain. This project downloads the plain-text
edition maintained by the [Spacelog](https://github.com/Spacelog/Spacelog)
project, pinned to a specific commit. Spacelog's repository does not state a
licence for its mission files. The full transcript is not stored in this
repository; `eval/results.json` and the cached drafts contain short
quotations from it.
