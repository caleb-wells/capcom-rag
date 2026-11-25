"""Turn retrieved passages into a grounded, cited answer.

The language model is the least trustworthy part of the pipeline, so it is
fenced in on three sides:

1. In the prompt, the model may use only the passages, must cite the
   timestamp of each line it relies on, and has an explicit way to decline.
2. After the call, code checks every cited timestamp against the passages the
   model was actually given. A citation that fails the check is flagged.
3. An answer with no verifiable citation at all is reported as unsupported.

A fourth fence was planned and removed: refusing up front when the best
passage's similarity fell below a floor. On this corpus the scores of
unanswerable questions overlap those of answerable ones almost completely
(see the README), so any floor either did nothing or refused good questions.

Generation talks to any OpenAI-compatible chat endpoint (LM Studio, Ollama,
llama.cpp, vLLM) using only the standard library. If no endpoint is
reachable, the answer falls back to the retrieved passages themselves.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .retrieve import Hit

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LLM_MODEL")  # default: the first chat model the server lists
API_KEY = os.environ.get("LLM_API_KEY", "not-needed-for-local-servers")

REFUSAL = "The transcript does not answer this."

SYSTEM_PROMPT = f"""\
You answer questions about the Apollo 11 mission using only excerpts from the \
air-to-ground voice transcript that the user provides.

Rules:
- Use only the excerpts. Do not use anything else you know about Apollo 11.
- After each claim, cite the timestamp of the transcript line that supports \
it, copied exactly, in square brackets, for example [04:06:46:04].
- Quote the transcript's own words when they are the answer.
- If the excerpts do not contain the answer, reply with exactly this sentence \
and nothing else: {REFUSAL}
- Keep the answer to a few sentences.

Speaker labels: CDR is Neil Armstrong (commander), CMP is Michael Collins \
(command module pilot), LMP is Buzz Aldrin (lunar module pilot), and CC is \
the capsule communicator in Houston. Timestamps are days:hours:minutes:seconds \
since launch."""

_CITATION = re.compile(r"\[(\d{2}:\d{2}:\d{2}:\d{2})\]")
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    source: str  # "llm" or "retrieval-only" (no model was reachable)
    refused: bool = False
    unsupported: bool = False  # answered, but with no citation that checks out
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


class LLMUnavailable(Exception):
    pass


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    # Passages go in mission order, not rank order: a conversation is easier
    # to follow chronologically, for a model as much as for a person.
    ordered = sorted(hits, key=lambda hit: hit.chunk.start_seconds)
    excerpts = "\n\n".join(
        f"<excerpt>\n{hit.chunk.display}\n</excerpt>" for hit in ordered
    )
    user = f"Transcript excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def check_citations(text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    """Split the timestamps cited in `text` into (valid, invalid).

    Valid means the timestamp is a real line in a passage the model was shown.
    This catches invented and misremembered citations. It does not prove the
    cited line supports the claim; that needs a human or a judge model.
    """
    shown = {timestamp for hit in hits for timestamp in hit.chunk.timestamps}
    cited = list(dict.fromkeys(_CITATION.findall(text)))
    return [c for c in cited if c in shown], [c for c in cited if c not in shown]


def answer(question: str, hits: list[Hit]) -> Answer:
    try:
        text = chat(build_messages(question, hits))
    except LLMUnavailable as error:
        note = f"(No language model reachable: {error}. Showing the retrieved passages.)"
        return Answer(note, hits, source="retrieval-only")

    if REFUSAL.rstrip(".").lower() in text.lower():
        return Answer(REFUSAL, hits, source="llm", refused=True)
    valid, invalid = check_citations(text, hits)
    return Answer(
        text, hits, source="llm", citations=valid, invalid_citations=invalid,
        unsupported=not valid,  # also covers a model that returned nothing
    )


# -- the HTTP call -------------------------------------------------------------


def _request(path: str, payload: dict | None = None, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise LLMUnavailable(f"{BASE_URL} did not respond ({error})") from error


def _default_model() -> str:
    models = [m["id"] for m in _request("/models", timeout=5).get("data", [])]
    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        raise LLMUnavailable(f"{BASE_URL} lists no chat model")
    return chat_models[0]


def chat(messages: list[dict]) -> str:
    payload = {
        "model": MODEL or _default_model(),
        "messages": messages,
        "temperature": 0,
        # Generous, because reasoning models spend tokens thinking before
        # they write the short answer.
        "max_tokens": 3000,
    }
    message = _request("/chat/completions", payload)["choices"][0]["message"]
    return _THINKING.sub("", message.get("content") or "").strip()
