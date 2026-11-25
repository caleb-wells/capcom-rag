"""Command line: fetch, index, ask, eval."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from . import corpus, evaluate, generate
from .retrieve import INDEX_DIR, MODES, Retriever


def _load_retriever() -> Retriever:
    if not (INDEX_DIR / "chunks.json").exists():
        sys.exit("No index found. Run `python -m capcom_rag index` first.")
    return Retriever.load()


def cmd_fetch(_: argparse.Namespace) -> None:
    path = corpus.fetch()
    print(f"Transcript ready at {path} ({path.stat().st_size / 1e6:.1f} MB, checksum verified)")


def cmd_index(_: argparse.Namespace) -> None:
    utterances = corpus.load()
    print(f"Parsed {len(utterances):,} utterances. Embedding chunks (a minute or two on CPU)...")
    retriever = Retriever.build(utterances)
    retriever.save()
    print(f"Indexed {len(retriever.chunks):,} chunks into {INDEX_DIR}/")


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = _load_retriever()
    hits = retriever.search(args.question, args.k, args.mode)
    result = generate.answer(args.question, hits)

    print(f"\n{result.text}\n")
    if result.invalid_citations:
        print(f"WARNING: cited but not in the retrieved passages: {result.invalid_citations}\n")
    elif result.citations:
        print(f"Citations verified against retrieved passages: {len(result.citations)}\n")

    if args.show_passages or result.source == "retrieval-only":
        for rank, hit in enumerate(hits, start=1):
            print(f"--- #{rank}  {hit.chunk.start} to {hit.chunk.end}  "
                  f"(score {hit.score:.3f}, similarity {hit.similarity:.2f})")
            for line in hit.chunk.display.splitlines():
                print(textwrap.fill(line, 100, subsequent_indent="      "))
            print()


def cmd_eval(args: argparse.Namespace) -> None:
    retriever = _load_retriever()
    questions = evaluate.load_questions()

    retrieval = evaluate.evaluate_retrieval(retriever, questions)
    print(f"Retrieval, {sum(bool(q['gold']) for q in questions)} answerable questions, "
          f"{len(retriever.chunks):,} chunks\n")
    print(evaluate.format_retrieval_table(retrieval))
    report = {"retrieval": retrieval}

    if not args.retrieval_only:
        try:
            generation = evaluate.evaluate_generation(retriever, questions)
        except generate.LLMUnavailable as error:
            print(f"\nSkipping generation eval: {error}")
        else:
            report["generation"] = generation
            g = generation
            print(f"""
Generation (hybrid+rerank, top 5)
  Answerable questions answered:      {g['answered']}/{g['answerable']}
    ...citing a gold timestamp:       {g['answered_citing_gold']}/{g['answered']}
    ...with no valid citation:        {g['answered_without_citation']}/{g['answered']}
  Unanswerable questions refused:     {g['correctly_refused']}/{g['unanswerable']}
  Citations pointing outside context: {g['invalid_citations']}/{g['citations']}""")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="capcom_rag", description=__doc__)
    commands = parser.add_subparsers(required=True)

    commands.add_parser("fetch", help="download the transcript").set_defaults(run=cmd_fetch)
    commands.add_parser("index", help="chunk, embed and save the index").set_defaults(run=cmd_index)

    ask = commands.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5, help="passages to retrieve (default 5)")
    ask.add_argument("--mode", choices=MODES, default="hybrid+rerank")
    ask.add_argument("--show-passages", action="store_true", help="print the retrieved passages")
    ask.set_defaults(run=cmd_ask)

    evaluation = commands.add_parser("eval", help="run the evaluation suite")
    evaluation.add_argument("--retrieval-only", action="store_true", help="skip the LLM part")
    evaluation.add_argument("--output", help="write the full JSON report here")
    evaluation.set_defaults(run=cmd_eval)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
