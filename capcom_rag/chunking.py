"""Group utterances into retrievable chunks.

A single utterance ("Roger.") means nothing on its own, so the unit of
retrieval is a short stretch of conversation. Three rules shape a chunk:

1. A word budget keeps chunks short. 160 words did better than 100 on the
   eval set; larger sizes were not tried.
2. A long silence ends the chunk early, because radio silence is the
   transcript's natural topic boundary.
3. Consecutive chunks share a few utterances, so an exchange that straddles a
   boundary is still intact in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Utterance

# The transcript identifies speakers by role and call sign. People ask about
# "Armstrong", a name that appears once in the whole transcript, so the
# searchable copy of each chunk swaps in surnames and opens with one line of
# context. The first version spelled the full description out on every line
# and scored worse on the eval set; see the README.
NAMES = {"CDR": "Armstrong", "CMP": "Collins", "LMP": "Aldrin", "CC": "Houston"}
ROLES = {
    "CDR": "Armstrong is the commander",
    "CMP": "Collins is the command module pilot",
    "LMP": "Aldrin is the lunar module pilot",
    "CC": "Houston is mission control",
}
CALL_SIGNS = {
    "EAGLE": "aboard the lunar module Eagle",
    "TRANQ": "landed on the Moon at Tranquility Base",
    "EVA": "walking on the lunar surface",
    "COLUMBIA": "aboard the command module Columbia in lunar orbit",
}


def speaker_name(label: str) -> str:
    """'CDR (EVA)' -> 'Armstrong'. Unknown labels pass through."""
    return NAMES.get(label.partition(" (")[0], label)


def context_line(labels: list[str]) -> str:
    """One sentence saying who is talking in this chunk, and from where."""
    facts = []
    for label in dict.fromkeys(labels):  # unique, in order of appearance
        role, _, call_sign = label.partition(" (")
        fact = ROLES.get(role)
        if fact and call_sign.rstrip(")") in CALL_SIGNS:
            fact += ", " + CALL_SIGNS[call_sign.rstrip(")")]
        if fact and not any(f.startswith(fact) for f in facts):
            facts = [f for f in facts if not fact.startswith(f)] + [fact]
    return "; ".join(facts) + "." if facts else ""


@dataclass(frozen=True)
class Chunk:
    id: int
    start: str  # timestamp of the first utterance
    end: str  # timestamp of the last utterance
    start_seconds: int
    end_seconds: int
    timestamps: tuple[str, ...]  # every utterance timestamp, for citation checks
    display: str  # timestamped lines, shown to the reader and the LLM
    search_text: str  # enriched lines without timestamps, for the indexes


def chunk_utterances(
    utterances: list[Utterance],
    max_words: int = 160,
    overlap: int = 2,
    max_gap_seconds: int = 300,
    enrich: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    window: list[Utterance] = []
    words = 0
    fresh = 0  # utterances in the window that no earlier chunk has covered

    def emit() -> None:
        chunks.append(_make_chunk(len(chunks), window, enrich))

    for utterance in utterances:
        gap = window and utterance.seconds - window[-1].seconds > max_gap_seconds
        full = window and words + _count(utterance) > max_words
        if gap or full:
            if fresh:
                emit()
            # Carry context across a size split, but never across a silence.
            window = [] if gap else window[-overlap:] if overlap else []
            words = sum(_count(u) for u in window)
            fresh = 0
        window.append(utterance)
        words += _count(utterance)
        fresh += 1
    if fresh:
        emit()
    return chunks


def _count(utterance: Utterance) -> int:
    return len(utterance.text.split()) + 1  # +1 for the speaker label


def _make_chunk(chunk_id: int, window: list[Utterance], enrich: bool) -> Chunk:
    if enrich:
        lines = [context_line([u.speaker for u in window])]
        lines += [f"{speaker_name(u.speaker)}: {u.text}" for u in window]
    else:
        lines = [f"{u.speaker}: {u.text}" for u in window]
    return Chunk(
        id=chunk_id,
        start=window[0].timestamp,
        end=window[-1].timestamp,
        start_seconds=window[0].seconds,
        end_seconds=window[-1].seconds,
        timestamps=tuple(u.timestamp for u in window),
        display="\n".join(f"[{u.timestamp}] {u.speaker}: {u.text}" for u in window),
        search_text="\n".join(line for line in lines if line),
    )
