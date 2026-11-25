"""Fast unit tests. None of them download a model or need the transcript."""

import numpy as np
import pytest

from capcom_rag import generate
from capcom_rag.bm25 import BM25
from capcom_rag.chunking import chunk_utterances, context_line, speaker_name
from capcom_rag.corpus import parse, to_seconds
from capcom_rag.evaluate import first_relevant_rank, retrieval_metrics
from capcom_rag.retrieve import Hit, reciprocal_rank_fusion

SAMPLE = """\
[04:06:45:59]
_page : 317
CDR (TRANQ): Houston, [glossary:Tranquility Base] here.

[04:06:46:04]
CDR (TRANQ): The Eagle has landed.

[04:06:46:06]
CC: Roger, Tranquility. We copy you on the ground. You got a bunch of guys
  about to turn blue. We're breathing again.

[04:08:00:00]
CMP (COLUMBIA): Sounds like it looks a lot better than it did yesterday.
CC: Roger.
"""


@pytest.fixture
def utterances():
    return parse(SAMPLE)


# -- parsing -----------------------------------------------------------------


def test_parse_reads_speakers_timestamps_and_continuations(utterances):
    assert [u.speaker for u in utterances] == [
        "CDR (TRANQ)", "CDR (TRANQ)", "CC", "CMP (COLUMBIA)", "CC",
    ]
    assert utterances[2].text.endswith("about to turn blue. We're breathing again.")
    assert utterances[3].timestamp == utterances[4].timestamp == "04:08:00:00"


def test_parse_strips_markup_and_metadata(utterances):
    assert utterances[0].text == "Houston, Tranquility Base here."
    assert not any("_page" in u.text for u in utterances)


def test_to_seconds():
    assert to_seconds("00:00:00:04") == 4
    assert to_seconds("04:06:46:04") == 4 * 86400 + 6 * 3600 + 46 * 60 + 4


# -- chunking ----------------------------------------------------------------


def test_chunks_never_span_a_long_silence(utterances):
    chunks = chunk_utterances(utterances, max_words=500, max_gap_seconds=300)
    assert [chunk.start for chunk in chunks] == ["04:06:45:59", "04:08:00:00"]
    assert "04:06:46:06" not in chunks[1].timestamps


def test_chunks_respect_word_budget_and_overlap(utterances):
    chunks = chunk_utterances(utterances[:3], max_words=12, overlap=1)
    # The two short lines fit the budget together; the long reply does not.
    assert [len(chunk.timestamps) for chunk in chunks] == [2, 2]
    assert chunks[1].timestamps[0] == chunks[0].timestamps[-1]


def test_every_utterance_lands_in_a_chunk(utterances):
    chunks = chunk_utterances(utterances, max_words=12, overlap=1)
    covered = {timestamp for chunk in chunks for timestamp in chunk.timestamps}
    assert covered == {u.timestamp for u in utterances}


def test_search_text_is_enriched_but_display_is_verbatim(utterances):
    chunk = chunk_utterances(utterances)[0]
    assert "Armstrong" in chunk.search_text and "Tranquility Base" in chunk.search_text
    assert "[04:06:46:04] CDR (TRANQ): The Eagle has landed." in chunk.display
    assert "Armstrong" not in chunk.display


def test_speaker_name_passes_unknown_labels_through():
    assert speaker_name("PRESIDENT NIXON") == "PRESIDENT NIXON"
    assert speaker_name("LMP (EVA)") == "Aldrin"


def test_context_line_describes_each_speaker_once():
    line = context_line(["CC", "LMP (EVA)", "CC", "LMP (EVA)", "SWIM 1"])
    assert line == (
        "Houston is mission control; "
        "Aldrin is the lunar module pilot, walking on the lunar surface."
    )


# -- retrieval ---------------------------------------------------------------


def test_bm25_prefers_rare_terms_and_ignores_unknown_ones():
    index = BM25(["roger roger roger", "program alarm 1202", "roger go for landing"])
    assert index.search("1202 roger", k=3)[0][0] == 1
    assert index.search("zebra", k=3) == []


def test_bm25_penalises_long_documents():
    index = BM25(["alarm", "alarm " + "filler " * 50])
    assert [doc for doc, _ in index.search("alarm", k=2)] == [0, 1]


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])
    order = [doc for doc, _ in fused]
    assert order[0] in (2, 3) and order[-1] == 9
    assert dict(fused)[2] == pytest.approx(2 / 62)


# -- generation --------------------------------------------------------------


def _hits(utterances, similarity=0.9):
    return [Hit(chunk, 1.0, similarity) for chunk in chunk_utterances(utterances)]


def test_check_citations_separates_real_from_invented(utterances):
    text = "They landed [04:06:46:04] and then went bowling [09:09:09:09]."
    assert generate.check_citations(text, _hits(utterances)) == (["04:06:46:04"], ["09:09:09:09"])


def test_prompt_orders_passages_chronologically(utterances):
    hits = list(reversed(_hits(utterances)))
    user = generate.build_messages("q", hits)[1]["content"]
    assert user.index("04:06:46:04") < user.index("04:08:00:00")


def test_answer_flags_invalid_citations(utterances, monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda messages: "Landed [04:06:46:04], maybe [01:01:01:01].")
    result = generate.answer("q", _hits(utterances))
    assert (result.citations, result.invalid_citations) == (["04:06:46:04"], ["01:01:01:01"])


def test_answer_without_a_verifiable_citation_is_unsupported(utterances, monkeypatch):
    for reply in ("They landed safely.", "They landed [01:01:01:01].", ""):
        monkeypatch.setattr(generate, "chat", lambda messages, reply=reply: reply)
        assert generate.answer("q", _hits(utterances)).unsupported


def test_answer_with_a_verified_citation_is_supported(utterances, monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda messages: "They landed [04:06:46:04].")
    assert not generate.answer("q", _hits(utterances)).unsupported


def test_answer_recognises_model_refusal(utterances, monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda messages: generate.REFUSAL)
    result = generate.answer("q", _hits(utterances))
    assert result.refused and result.source == "llm"


def test_answer_falls_back_when_no_model_is_running(utterances, monkeypatch):
    def unavailable(messages):
        raise generate.LLMUnavailable("connection refused")

    monkeypatch.setattr(generate, "chat", unavailable)
    result = generate.answer("q", _hits(utterances))
    assert result.source == "retrieval-only" and not result.refused


# -- evaluation --------------------------------------------------------------


def test_first_relevant_rank(utterances):
    hits = _hits(utterances)
    assert first_relevant_rank(hits, ["04:08:00:00"]) == 2
    assert first_relevant_rank(hits, ["00:00:00:00"]) is None


def test_retrieval_metrics():
    metrics = retrieval_metrics([1, 2, None, 8])
    assert metrics["recall@1"] == 0.25
    assert metrics["recall@5"] == 0.5
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 8) / 4)


def test_vectors_are_searched_by_dot_product():
    vectors = np.eye(3, dtype=np.float32)
    assert int(np.argmax(vectors @ np.array([0, 1, 0], dtype=np.float32))) == 1
