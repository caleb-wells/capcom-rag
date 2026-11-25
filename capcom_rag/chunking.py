"""Group utterances into retrievable chunks.

A single utterance ("Roger.") means nothing on its own, so the unit of
retrieval is a short stretch of conversation. Three rules shape a chunk:

1. A word budget keeps chunks small enough to embed well and cheap to prompt.
2. A long silence ends the chunk early, because radio silence is the
   transcript's natural topic boundary.
3. Consecutive chunks share a few utterances, so an exchange that straddles a
   boundary is still intact in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Utterance

# The transcript identifies speakers by role and call sign. People ask about
# "Armstrong" and "the moonwalk", which appear nowhere in the text, so the
# searchable copy of each line spells the label out. Index-time enrichment
# like this is cheap, and measurably the biggest single win in this repo.
ROLES = {
    "CDR": "Armstrong, commander",
    "CMP": "Collins, command module pilot",
    "LMP": "Aldrin, lunar module pilot",
    "CC": "Houston, mission control",
}
CALL_SIGNS = {
    "EAGLE": "aboard the lunar module Eagle",
    "TRANQ": "landed on the Moon at Tranquility Base",
    "EVA": "walking on the lunar surface",
    "COLUMBIA": "aboard the command module Columbia in lunar orbit",
}


def describe_speaker(label: str) -> str:
    """'CDR (EVA)' -> 'Armstrong, commander, walking on the lunar surface'."""
    role, _, call_sign = label.partition(" (")
    parts = [ROLES.get(role, role)]
    if call_sign.rstrip(")") in CALL_SIGNS:
        parts.append(CALL_SIGNS[call_sign.rstrip(")")])
    return ", ".join(parts)


@dataclass(frozen=True)
class Chunk:
    id: int
    start: str  # timestamp of the first utterance
    end: str  # timestamp of the last utterance
    start_seconds: int
    end_seconds: int
    timestamps: tuple[str, ...]  # every utterance timestamp, for citation checks
    display: str  # timestamped lines, shown to the reader and the LLM
    search_text: str  # the same lines without timestamps, for the indexes


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
    name = describe_speaker if enrich else str
    return Chunk(
        id=chunk_id,
        start=window[0].timestamp,
        end=window[-1].timestamp,
        start_seconds=window[0].seconds,
        end_seconds=window[-1].seconds,
        timestamps=tuple(u.timestamp for u in window),
        display="\n".join(f"[{u.timestamp}] {u.speaker}: {u.text}" for u in window),
        search_text="\n".join(f"{name(u.speaker)}: {u.text}" for u in window),
    )
