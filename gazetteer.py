"""Data-driven place matching and dictionary junk-detection.

Replaces the hand-maintained blacklists / spacing fixes with lookups against
static data files (see build_data.py). Pure stdlib (difflib) — no new deps.
"""
from __future__ import annotations

import re
import difflib
from pathlib import Path
from functools import lru_cache

DATA = Path(__file__).parent / "data"


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    p = DATA / "id_words.txt"
    if not p.exists():
        return frozenset()
    return frozenset(w.strip().lower()
                     for w in p.read_text(encoding="utf-8").splitlines()
                     if w.strip())


@lru_cache(maxsize=1)
def _places() -> tuple[dict[str, str], list[str]]:
    """Return ({despaced_key: canonical}, [despaced_keys]) for places+offices."""
    canon: dict[str, str] = {}
    for fn in ("id_places.txt", "id_offices_extra.txt"):
        p = DATA / fn
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            name = line.strip().upper()
            if name:
                canon.setdefault(_norm(name), name)
    return canon, list(canon.keys())


def is_dictionary_word(text: str) -> bool:
    """True if the whole token is a common Indonesian word (i.e. a label /
    notes-page word, never a name or place). Proper nouns are not in the list."""
    t = re.sub(r"[^a-z]", "", text.lower())
    return len(t) >= 3 and t in _words()


def match_place(text: str, cutoff: float = 0.86) -> str | None:
    """Snap an OCR'd value to a known Indonesian place/office name.
    Handles dropped spaces ('JAKARTATIMUR') and small OCR slips via fuzzy
    match. Returns the canonical name, or None if nothing is close enough."""
    if not text:
        return None
    key = _norm(text)
    if len(key) < 3:
        return None
    canon, keys = _places()
    if key in canon:                       # exact / despaced hit
        return canon[key]
    hit = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
    return canon[hit[0]] if hit else None
