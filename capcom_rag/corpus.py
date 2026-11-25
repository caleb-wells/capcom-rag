"""Fetch and parse the Apollo 11 air-to-ground transcript.

The transcript is a NASA work in the public domain. We use the plain-text
edition maintained by the Spacelog project, pinned to one commit
and verified by checksum so that the eval numbers are reproducible.
"""

from __future__ import annotations

import hashlib
import re
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

SOURCE_COMMIT = "305a06ec50fbf1f8d24a784b7d835578229057f6"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Spacelog/Spacelog/"
    f"{SOURCE_COMMIT}/missions/a11/transcripts/TEC"
)
SOURCE_SHA256 = "9c9d675da330847322c3bde18991a034d6ea85efc8ecebdf81ca2fb2a4713b2c"
DEFAULT_PATH = Path("data/apollo11_tec.txt")

_TIMESTAMP = re.compile(r"^\[(-?\d{2}:\d{2}:\d{2}:\d{2})\]\s*$")
_SPEAKER = re.compile(r"^([A-Z0-9/ ()-]+): (.*)$")
_GLOSSARY = re.compile(r"\[glossary:([^\]|]+)(?:\|[^\]]*)?\]")


@dataclass(frozen=True)
class Utterance:
    """One line of dialogue. `seconds` is ground elapsed time since launch."""

    timestamp: str  # "DD:HH:MM:SS", the citation key
    seconds: int
    speaker: str
    text: str


def to_seconds(timestamp: str) -> int:
    sign = -1 if timestamp.startswith("-") else 1
    d, h, m, s = (int(part) for part in timestamp.lstrip("-").split(":"))
    return sign * (d * 86400 + h * 3600 + m * 60 + s)


def fetch(path: Path = DEFAULT_PATH) -> Path:
    """Download the transcript if it is missing, and verify its checksum."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use certifi's CA bundle: the python.org build on macOS that this was
        # developed on failed certificate verification with the default one.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(SOURCE_URL, timeout=60, context=context) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{path} does not match the pinned transcript (sha256 {digest}). "
            "Delete it and run `fetch` again."
        )
    return path


def parse(raw: str) -> list[Utterance]:
    """Turn the transcript text into a flat list of utterances.

    The format is a timestamp line, optional `_meta` lines, then one or more
    `SPEAKER: text` lines. Indented lines continue the previous utterance.
    """
    utterances: list[Utterance] = []
    timestamp: str | None = None
    speaker: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal speaker, parts
        if timestamp and speaker and parts:
            text = _GLOSSARY.sub(r"\1", " ".join(parts))
            text = re.sub(r"\s+", " ", text).strip()
            utterances.append(Utterance(timestamp, to_seconds(timestamp), speaker, text))
        speaker, parts = None, []

    for line in raw.splitlines():
        if not line.strip() or line.startswith("_"):
            continue
        if match := _TIMESTAMP.match(line):
            flush()
            timestamp = match.group(1)
        elif match := _SPEAKER.match(line):
            flush()
            speaker, parts = match.group(1), [match.group(2)]
        elif speaker:
            parts.append(line.strip())
    flush()
    return utterances


def load(path: Path = DEFAULT_PATH) -> list[Utterance]:
    return parse(fetch(path).read_text(encoding="utf-8"))
