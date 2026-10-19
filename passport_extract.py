"""
Indonesian passport data extractor.

Strategy
--------
1. Run RapidOCR over the whole page to get text boxes + positions.
2. Locate the TD3 machine-readable zone (2 x 44-char lines) and parse it.
   The MRZ is check-digit validated and is the reliable source for:
   type, country code, passport no, name, nationality, date of birth,
   sex and date of expiry.
3. Three fields are NOT encoded in the MRZ and only exist in the visual
   inspection zone, so we read them by anchoring on their printed labels:
   - tanggal terbit / date of issue
   - tempat lahir   / place of birth
   - kantor         / issuing office
4. Prefer the human-readable visual dates when available (unambiguous
   century), fall back to MRZ-derived dates otherwise.

Output field names follow the request (Indonesian).
"""
from __future__ import annotations

import re
import os
import sys
import json
import csv
import queue
import argparse
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from gazetteer import match_place, is_dictionary_word

# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------
# RapidOCR wraps three ONNX sessions + Python pre/post-processing and is not
# guaranteed thread-safe, so a single instance must not be shared across
# threads. For single-threaded use (CLI, one process) we keep a lazy singleton.
# For multithreaded serving, use EnginePool (below): each thread borrows its own
# instance. Both are OS-agnostic (no process forking).
_ENGINE: RapidOCR | None = None
_ENGINE_LOCK = threading.Lock()


def _make_engine() -> RapidOCR:
    """Create a RapidOCR engine, honouring PPX_OCR_THREADS. Set it to 1 when
    running many worker processes so they don't each grab every core (ONNX
    defaults to all cores per process -> N processes oversubscribe the CPU)."""
    n = int(os.getenv("PPX_OCR_THREADS", "-1"))   # -1 = use all cores
    if n > 0:
        return RapidOCR(**{"Global.intra_op_num_threads": n,
                           "Global.inter_op_num_threads": n})
    return RapidOCR()


def _engine() -> RapidOCR:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = _make_engine()
    return _ENGINE


class EnginePool:
    """A fixed pool of RapidOCR instances borrowed via a thread-safe queue —
    real multithreaded OCR in one process, cross-platform, no forking.

    Note: RapidOCR/ONNXRuntime already parallelises each inference across cores.
    Running a large pool on top of that oversubscribes the CPU; keep the pool
    small (2-4) or lower ORT threads. See README."""

    def __init__(self, size: int = 2):
        self._q: "queue.Queue[RapidOCR]" = queue.Queue()
        self.size = max(1, size)
        for _ in range(self.size):
            self._q.put(_make_engine())

    @contextmanager
    def borrow(self):
        eng = self._q.get()
        try:
            yield eng
        finally:
            self._q.put(eng)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    # Indonesian variants sometimes appear
    "MEI": 5, "AGU": 8, "OKT": 10, "DES": 12,
}
MONTH_ABBR = ["", "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

DATE_RE = re.compile(r"(\d{1,2})\s*([A-Z]{3})\s*(\d{4})")


class Box:
    __slots__ = ("text", "score", "x", "y", "x0", "y0", "x1", "y1")

    def __init__(self, poly, text, score):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        self.text = text
        self.score = float(score)
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        self.x = (self.x0 + self.x1) / 2
        self.y = (self.y0 + self.y1) / 2


def _ocr(img, engine: RapidOCR | None = None) -> list[Box]:
    """OCR a numpy BGR image -> list[Box], using the given engine (or the
    module singleton). Pass a borrowed pool engine for thread-safe serving."""
    result, _ = (engine or _engine())(img)
    if not result:
        return []
    return [Box(poly, txt, sc) for poly, txt, sc in result]


# RapidOCR's detector resizes internally to ~1 kpx; feeding it a 3-4 kpx scan
# just wastes time. Cap the long edge — MRZ/text stay legible and OCR is ~3-4x
# faster, which matters most for per-upload latency in the web feature.
# Override with env PPX_MAX_EDGE.
_MAX_EDGE = int(os.getenv("PPX_MAX_EDGE", "1800"))


def _downscale(img: np.ndarray) -> np.ndarray:
    m = max(img.shape[:2])
    if m > _MAX_EDGE:
        s = _MAX_EDGE / m
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _decode(data: bytes) -> np.ndarray:
    """Decode image bytes in memory (no temp file — keeps passport PII off disk)."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image (unsupported or corrupt file)")
    return _downscale(img)


def _imread(path: str) -> np.ndarray:
    """Unicode-safe image read (cv2.imread fails on non-ASCII Windows paths)."""
    return _decode(np.fromfile(path, dtype=np.uint8).tobytes())


def _load_image(src) -> np.ndarray:
    """Accept a file path (str/Path), raw bytes, or an ndarray -> BGR image."""
    if isinstance(src, np.ndarray):
        return _downscale(src)
    if isinstance(src, (bytes, bytearray)):
        return _decode(bytes(src))
    return _imread(str(src))


def _rotate(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _mrz_score(boxes: list[Box]) -> tuple[int, dict | None]:
    """Return (quality, parsed) for the best MRZ found in these boxes.
    quality: 2 = MRZ found and all check digits valid, 1 = found, 0 = none."""
    lines = _find_mrz_lines(boxes)
    if not lines:
        return 0, None
    parsed = parse_mrz(*lines)
    v = parsed.get("_mrz_valid", {})
    good = sum(1 for ok in v.values() if ok)
    return (2 if good >= 2 else 1), parsed


def _horizontality(boxes: list[Box]) -> int:
    """Number of boxes whose text line is wider than tall. RapidOCR's angle
    classifier reads word CONTENT at any rotation, so word-based signals can't
    tell orientation apart — but box shape can: upright/upside-down pages have
    horizontal (wide) text lines, 90/270-rotated pages have vertical (tall) ones."""
    return sum(1 for b in boxes if (b.x1 - b.x0) > (b.y1 - b.y0))


def _best_orientation(img: np.ndarray, engine: RapidOCR | None = None):
    """Return (visual_boxes, mrz_parsed, angle) for an already-loaded image.

    The printed ("visual") fields need the upright orientation, but RapidOCR
    sometimes drops an MRZ line when the page is upright yet reads it when the
    page is rotated. So we choose orientation independently for each: visual
    from the most-horizontal orientation (with a prior toward upright, since
    almost all uploads are), MRZ from whichever orientation parses a valid one.
    """
    results = []  # (angle, boxes, parsed, q, horiz)

    def _consider(angle):
        boxes = _ocr(_rotate(img, angle), engine)
        q, parsed = _mrz_score(boxes)
        h = _horizontality(boxes)
        results.append((angle, boxes, parsed, q, h))
        return h

    # Try 0, then the rotations, stopping as soon as we hold both a usable
    # visual orientation (some horizontal page) and a check-digit-valid MRZ.
    # 180 is only attempted as a last resort (upside-down uploads are rare).
    for angle in (0, 90, 270):
        h = _consider(angle)
        if angle == 0 and results[-1][3] >= 2 and h >= 6:
            return results[-1][1], results[-1][2], 0     # fast path: 1 OCR pass
        have_visual = any(r[4] >= 6 for r in results)
        have_mrz = any(r[3] >= 2 for r in results)
        if have_visual and have_mrz:
            break
    else:
        if not any(r[4] >= 6 for r in results):          # nothing horizontal yet
            _consider(180)

    def vscore(r):                       # +bias for angle 0 to break 0/180 ties
        return r[4] + (5 if r[0] == 0 else 0)

    up_best = max(results, key=vscore)                       # visual orientation
    mrz_best = max(results, key=lambda r: (r[3], vscore(r)))  # MRZ orientation
    parsed = mrz_best[2] if mrz_best[3] >= 1 else up_best[2]
    return up_best[1], parsed, up_best[0]


# ---------------------------------------------------------------------------
# MRZ (TD3) parsing
# ---------------------------------------------------------------------------
def _clean_mrz(s: str) -> str:
    return re.sub(r"[^A-Z0-9<]", "", s.upper().replace(" ", ""))


def _mrz_top(boxes: list[Box]) -> float:
    """Top y of the MRZ block (for bounding the visual zone)."""
    ys = []
    for b in boxes:
        t = _clean_mrz(b.text)
        if len(t) >= 25 and (t.count("<") >= 2 or _looks_like_mrz_l2(t)):
            ys.append(b.y0)
    return min(ys) if ys else 1e9


def _looks_like_mrz_l2(t: str) -> bool:
    """TD3 line 2 is ~44 chars of A-Z0-9< with mostly digits."""
    if len(t) < 25:
        return False
    digits = sum(c.isdigit() for c in t)
    return digits >= 8 and all(c.isalnum() or c == "<" for c in t)


def _find_mrz_lines(boxes: list[Box]) -> tuple[str, str] | None:
    """Find the two TD3 lines. They are told apart by digit content, not by
    '<' runs: line 1 is letter-heavy (name + '<<' separator, ~no digits),
    line 2 is digit-heavy (passport no, dates, check digits) and may itself
    contain '<<' filler runs."""
    cleaned = [(b.y, _clean_mrz(b.text)) for b in boxes]
    l2c = sorted((y, t) for y, t in cleaned if _looks_like_mrz_l2(t))
    l1c = sorted((y, t) for y, t in cleaned
                 if "<<" in t and len(t) >= 20
                 and sum(c.isdigit() for c in t) < 8)
    if not l2c or not l1c:
        return None
    y2, line2 = l2c[-1]                       # bottom-most digit line
    above = [(y, t) for y, t in l1c if y < y2 + 5]  # line 1 sits above line 2
    _, line1 = above[-1] if above else l1c[-1]
    line1 = (line1 + "<" * 44)[:44]
    line2 = (line2 + "<" * 44)[:44]
    return line1, line2


def _mrz_check(data: str) -> int:
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        if ch == "<":
            v = 0
        elif ch.isdigit():
            v = int(ch)
        else:
            v = ord(ch) - 55  # A=10..Z=35
        total += v * weights[i % 3]
    return total % 10


def _fix_digits(s: str) -> str:
    """Common OCR letter->digit fixes for numeric MRZ fields."""
    return (s.replace("O", "0").replace("Q", "0").replace("D", "0")
             .replace("I", "1").replace("L", "1").replace("Z", "2")
             .replace("S", "5").replace("B", "8").replace("G", "6"))


# OCR characters that get confused both ways; used to repair a field whose MRZ
# check digit fails by trying the plausible substitutions until it passes.
_CONFUSIONS = {
    "0": "O", "O": "0", "1": "I", "I": "1", "2": "Z", "Z": "2",
    "5": "S", "S": "5", "8": "B", "B": "8", "6": "G", "G": "6",
    "D": "0", "Q": "0",
}


def _correct_field(field: str, check: str) -> tuple[str, bool]:
    """Repair an MRZ field so its check digit validates, trying single- then
    two-character OCR substitutions. Returns (field, validated). ICAO check
    digits make a passing correction almost always the true value."""
    if not check.isdigit():
        return field, False
    target = int(check)
    if _mrz_check(field) == target:
        return field, True
    n = len(field)
    # single-character fixes first (most common)
    for i in range(n):
        c = field[i]
        alt = _CONFUSIONS.get(c)
        if not alt:
            continue
        cand = field[:i] + alt + field[i + 1:]
        if _mrz_check(cand) == target:
            return cand, True
    # then pairs (cheap: fields are short)
    for i in range(n):
        ai = _CONFUSIONS.get(field[i])
        if not ai:
            continue
        for j in range(i + 1, n):
            aj = _CONFUSIONS.get(field[j])
            if not aj:
                continue
            cand = field[:i] + ai + field[i + 1:j] + aj + field[j + 1:]
            if _mrz_check(cand) == target:
                return cand, True
    return field, False


def _yymmdd_to_date(s: str, is_dob: bool) -> str | None:
    s = _fix_digits(s)
    if not re.fullmatch(r"\d{6}", s):
        return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    now = datetime.now().year % 100
    if is_dob:
        year = 2000 + yy if yy <= now else 1900 + yy
    else:  # expiry -> assume within +/- some window, lean future
        year = 2000 + yy
    return f"{dd:02d} {MONTH_ABBR[mm]} {year}"


def _clean_name(part: str) -> str:
    """Normalise an MRZ name component: '<' filler -> space, drop the lone 'K'
    tokens that OCR produces from the '<' separators."""
    s = re.sub(r"\s+", " ", part.replace("<", " ")).strip()
    # a standalone 'K' in an Indonesian passport name is virtually always a
    # mis-read '<' filler, not a real name part
    s = re.sub(r"\bK\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _name_looks_bad(name: str) -> bool:
    """MRZ name is unreliable if it carries digits or is implausibly short."""
    if not name:
        return True
    if any(c.isdigit() for c in name):
        return True
    return len(re.sub(r"[^A-Za-z]", "", name)) < 4


def parse_mrz(line1: str, line2: str) -> dict:
    out: dict = {"_mrz_valid": {}}
    # Line 1: P<CCCSURNAME<<GIVEN<NAMES<<<
    out["jenis_type"] = line1[0] if line1[0] != "<" else ""
    country_l1 = _clean_mrz(line1[2:5]).replace("<", "")
    names = line1[5:]
    if "<<" in names:
        surname, given = names.split("<<", 1)
    else:
        surname, given = names, ""
    surname = _clean_name(surname)
    given = _clean_name(given)
    out["nama_lengkap"] = re.sub(r"\s+", " ", (given + " " + surname)).strip()

    # Line 2 — repair each field against its ICAO check digit where possible
    doc_field, doc_ok = _correct_field(line2[0:9], line2[9])
    nationality = _clean_mrz(line2[10:13]).replace("<", "")
    dob_field, dob_ok = _correct_field(line2[13:19], line2[19])
    sex = line2[20]
    exp_field, exp_ok = _correct_field(line2[21:27], line2[27])

    passport_no = doc_field.replace("<", "")
    out["no_paspor"] = _fix_digits(passport_no) if passport_no.isdigit() else passport_no
    out["_mrz_valid"]["no_paspor"] = doc_ok
    out["kode_negara"] = _fix_country(country_l1, nationality)
    out["kewarganegaraan"] = _fix_country(nationality, country_l1)
    out["tanggal_lahir"] = _yymmdd_to_date(dob_field, is_dob=True)
    out["_mrz_valid"]["tanggal_lahir"] = dob_ok
    out["sex"] = {"M": "M", "F": "F"}.get(sex, sex if sex != "<" else "")
    out["tanggal_habis_berlaku"] = _yymmdd_to_date(exp_field, is_dob=False)
    out["_mrz_valid"]["tanggal_habis_berlaku"] = exp_ok
    return out


# ---------------------------------------------------------------------------
# Visual inspection zone (label-anchored) for fields not in MRZ
# ---------------------------------------------------------------------------
def _norm(t: str) -> str:
    return re.sub(r"[^A-Z]", "", t.upper())


def _name_key(t: str) -> str:
    """Normalised name for matching, ignoring the stray 'K' that OCR fuses in
    from MRZ '<' fillers (so 'IMELDAKRAHMI' matches 'IMELDARAHMI')."""
    return _norm(t).replace("K", "")


# The fixed, closed vocabulary printed as field labels on every Indonesian
# passport (ISO/ICAO template) — Indonesian + the standard English labels.
# This is a spec constant, not an open-ended blacklist: it never grows, so it is
# correct to hardcode. Matched as substrings to survive OCR word-merges
# ("NAMALENGKAP", "ISSUINGOFFICE"). "OFEICE" is the common OCR mangle of OFFICE.
_LABEL_TOKENS = [
    # Indonesian
    "JENIS", "KODE", "NEGARA", "PASPOR", "NAMA", "LENGKAP", "WARGANEGARA",
    "KEWARGANEGARAAN", "LAHIR", "KELAMIN", "TEMPAT", "PENGELUARAN", "HABIS",
    "BERLAKU", "MENGELUARKAN", "REPUBLIK", "INDONESIA", "TGL", "KANTOR", "YANG",
    # English (the 19 spec words)
    "TYPE", "COUNTRY", "CODE", "PASSPORT", "FULL", "NAME", "NATIONALITY",
    "BIRTH", "DATE", "SEX", "PLACE", "ISSUE", "EXPIRY", "ISSUING", "OFFICE",
    "OFEICE", "REPUBLIC", "REG",
]


# Known mobile-scanner watermarks (a fixed set of app signatures). These are
# stamped by the scanning app, not part of the passport — never a field value,
# and their presence is informational (a re-scan hint), not a data problem.
_SCANNER_TOKENS = (
    "TAPSCANNER", "CAMSCANNER", "SCANNEDBY", "SCANNEDWITH", "ADOBESCAN",
    "OFFICELENS", "MICROSOFTLENS", "GENIUSSCAN", "CLEARSCANNER", "FASTSCANNER",
    "SCANBOT", "DOCSCANNER", "SCANNERAPP", "MADEWITHSCAN", "SIMPLESCAN",
    "CREATEDBYSCAN", "VFLAT", "KAGISCAN", "PHOTOSCAN",
)


def _is_scanner_watermark(t: str) -> bool:
    n = _norm(t)
    return any(tok in n for tok in _SCANNER_TOKENS)


def _is_junk_value(t: str) -> bool:
    """Structural rejects for a would-be value box. Content-based rejection
    (dictionary words, non-place text) is handled by the gazetteer, not here."""
    n = _norm(t)
    if not n or len(n) < 2:
        return True
    if _is_scanner_watermark(t):       # android scanner stamp, not passport text
        return True
    if ":" in t or "/" in t:           # "KEPADA:", sex markers "P/F"/"L/M"
        return True
    if "<" in t:                       # a stray MRZ line
        return True
    if any(c.isdigit() for c in t):    # reg-no leak; place names have no digits
        return True
    # a common Indonesian word is a label / notes word, UNLESS it is also a real
    # place name (e.g. many cities are dictionary words too)
    if is_dictionary_word(t) and not match_place(t):
        return True
    return False


def _fix_country(*codes: str) -> str:
    """Indonesian passports: normalise the 3-letter code, tolerating OCR
    slips like 1DN / IDM / 1D2. Falls back to the first non-empty code."""
    for c in codes:
        cc = re.sub(r"[^A-Z0-9]", "", (c or "").upper())
        if cc == "IDN" or "DN" in cc or (len(cc) == 3 and cc[0] in "I1"
                                         and cc.endswith("N")):
            return "IDN"
    for c in codes:
        cc = re.sub(r"[^A-Z0-9]", "", (c or "").upper())
        if cc:
            return cc
    return ""


def _is_label_text(t: str) -> bool:
    n = _norm(t)
    if not n:
        return True
    # a slash strongly signals a bilingual label like "TGL LAHIR / DATE"
    if "/" in t and len(n) > 6:
        return True
    hits = sum(1 for tok in _LABEL_TOKENS if tok in n)
    # value words like SUKABUMI/BANDUNG won't hit these tokens
    return hits >= 1 and n not in ("INDONESIA",)


def _find_label(boxes: list[Box], keywords: list[str]) -> Box | None:
    best = None
    for b in boxes:
        n = _norm(b.text)
        if any(k in n for k in keywords):
            if best is None or b.y < best.y:
                best = b
    return best


def _values_below(boxes: list[Box], label: Box, mrz_top: float,
                  want_date: bool, exclude: set[str] | None = None,
                  raw: bool = False) -> list[str]:
    """Ranked (nearest-first) value strings printed under a label, above the
    MRZ. `exclude` rejects normalised strings (e.g. the holder name).
    `raw` returns every non-label value verbatim (for fields like the passport
    number that legitimately contain digits)."""
    exclude = exclude or set()
    exkeys = {_name_key(e) for e in exclude}
    cands = []
    for b in boxes:
        if b is label:
            continue
        if b.y <= label.y + 3:          # must be below label
            continue
        if b.y0 >= mrz_top - 5:         # above MRZ
            continue
        # horizontal overlap with label column
        overlap = min(b.x1, label.x1) - max(b.x0, label.x0)
        near = overlap > -30 or abs(b.x0 - label.x0) < 60
        if not near:
            continue
        # skip boxes that are themselves printed labels
        if _is_label_text(b.text) and not DATE_RE.search(b.text.upper()):
            continue
        dy = b.y - label.y
        cands.append((dy, b))
    cands.sort(key=lambda c: c[0])
    out = []
    for _, b in cands:
        txt = b.text.strip()
        if raw:
            if txt and not _is_label_text(txt):
                out.append(txt.upper())
            continue
        if want_date:
            d = _extract_date(txt)
            if d:
                out.append(d)
            continue
        if _extract_date(txt) or _is_junk_value(txt):
            continue
        if _norm(txt) in exclude or _name_key(txt) in exkeys:
            continue
        if re.search(r"[A-Za-z]", txt) and len(_norm(txt)) >= 2:
            out.append(txt.upper())
    return out


def _value_below(boxes: list[Box], label: Box, mrz_top: float,
                 want_date: bool, exclude: set[str] | None = None) -> str | None:
    vals = _values_below(boxes, label, mrz_top, want_date, exclude)
    return vals[0] if vals else None


def _pick_place(candidates: list[str]) -> tuple[str | None, bool]:
    """From ranked candidates, prefer the first that resolves to a known place
    (verified); otherwise keep the first plausible raw value (unverified)."""
    raw_fallback = None
    for c in candidates:
        val, ok = _resolve_place(c)
        if ok:
            return val, True
        if raw_fallback is None:
            raw_fallback = val
    return raw_fallback, False


def _resolve_place(v: str | None) -> tuple[str | None, bool]:
    """Snap a raw place/office value to the gazetteer.
    Returns (value, verified). Overseas KBRI/KJRI missions are accepted by
    prefix. Unmatched values are kept raw but reported as unverified."""
    if not v:
        return v, False
    raw = re.sub(r"\s+", " ", v.strip().rstrip(".")).upper()
    canonical = match_place(raw)
    if canonical:
        return canonical, True
    if re.match(r"^(KBRI|KJRI|KDEI)\b", raw) or raw == "LUAR NEGERI":
        return raw, True
    return raw, False


# Day/year digits may carry OCR letter-confusions (O->0, l/I->1, S->5, ...), so
# the numeric groups allow those letters and are repaired via _fix_digits. The
# 3-letter month anchor keeps this from matching arbitrary text.
_DATE_LOOSE = re.compile(r"([0-9OILZSBG]{1,2})\s*([A-Z]{3})\s*([0-9OILZSBG]{4})")


def _extract_date(txt: str) -> str | None:
    """Parse a printed date, tolerating common OCR slips in the digits and
    Indonesian month spellings; returns canonical 'DD MON YYYY' or None."""
    m = _DATE_LOOSE.search(txt.upper())
    if not m:
        return None
    day_s, mon, year_s = _fix_digits(m.group(1)), m.group(2)[:3], _fix_digits(m.group(3))
    if not (day_s.isdigit() and year_s.isdigit() and mon in MONTHS):
        return None
    day, year = int(day_s), int(year_s)
    if not (1 <= day <= 31 and 1900 <= year <= 2100):
        return None
    return f"{day:02d} {MONTH_ABBR[MONTHS[mon]]} {year}"


def _parse_date_text(txt: str) -> str | None:
    return _extract_date(txt)


def _printed_name(boxes: list[Box], mrz_top: float) -> str | None:
    """Read the printed 'Nama Lengkap / Full Name' value under its label."""
    lbl = _find_label(boxes, ["NAMALENGKAP", "FULLNAME", "NAMALENG"])
    if not lbl:
        return None
    val = _value_below(boxes, lbl, mrz_top, want_date=False)
    return val


# --- printed-zone fallback for the fields normally taken from the MRZ --------
# Used when the MRZ is unreadable, so a clean scan is still fully extracted
# (with no check digits -> the record is still flagged for review).
def _printed_passport_no(boxes: list[Box], mrz_top: float) -> str | None:
    """Indonesian passport no. is 1-2 leading letters + 6-8 digits. OCR slips in
    the numeric tail (O->0, S->5, ...) are repaired via _fix_digits."""
    lbl = _find_label(boxes, ["NOPASPOR", "PASSPORTNO", "PASPORNO"])
    if not lbl:
        return None
    for v in _values_below(boxes, lbl, mrz_top, want_date=False, raw=True):
        t = re.sub(r"[^A-Z0-9]", "", v.upper())   # keep digits (unlike _norm)
        if not (7 <= len(t) <= 9):
            continue
        prefix, tail = re.match(r"([A-Z]{0,2})(.*)", t).groups()
        cand = prefix + _fix_digits(tail)
        if re.fullmatch(r"[A-Z]{0,2}\d{6,8}", cand):
            return cand
    return None


def _printed_dob(boxes: list[Box], mrz_top: float) -> str | None:
    lbl = _find_label(boxes, ["TGLLAHIR", "DATEOFBIRTH"])
    if not lbl:
        return None
    return _value_below(boxes, lbl, mrz_top, want_date=True)


def _norm_sex(v: str) -> str | None:
    """Map a printed sex value to M/F, tolerating OCR slips in the separator and
    stray chars: the field reads 'L/M' (laki=male) or 'P/F' (perempuan=female),
    so 'L/M','LM','L1M','LiM','L|M','LI MM','M','L' -> M; 'P/F','PF','P','F' -> F.
    Decided by which marker set is present; ambiguous -> None (flagged)."""
    s = re.sub(r"[^A-Za-z]", "", v).upper()
    if not s:
        return None
    male = ("L" in s) or ("M" in s)
    female = ("P" in s) or ("F" in s)
    if male and not female:
        return "M"
    if female and not male:
        return "F"
    return None


def _printed_sex(boxes: list[Box], mrz_top: float) -> str | None:
    # "ELAMIN" also matches OCR K-slips seen in the dump (RELAMIN/XELAMIN)
    lbl = _find_label(boxes, ["ELAMIN", "SEX"])
    if not lbl:
        return None
    for v in _values_below(boxes, lbl, mrz_top, want_date=False, raw=True):
        sx = _norm_sex(v)
        if sx:
            return sx
    return None


def extract_visual(boxes: list[Box], mrz_top: float, page_w: float,
                   name: str = "") -> dict:
    out = {}
    midx = page_w / 2
    name_n = _norm(name)

    def _is_name(b: Box) -> bool:
        return bool(name_n) and (_norm(b.text) == name_n
                                 or _name_key(b.text) == _name_key(name))

    # candidate value boxes: above MRZ, not labels, skip header band
    top_cut = mrz_top * 0.10  # ignore very top (REPUBLIK INDONESIA header)
    vals = [b for b in boxes
            if b.y0 < mrz_top - 5 and b.y > top_cut
            and not _is_label_text(b.text)]
    left = sorted([b for b in vals if b.x < midx], key=lambda b: b.y)
    right = sorted([b for b in vals if b.x >= midx], key=lambda b: b.y)

    left_dates = [b for b in left if _parse_date_text(b.text)]
    right_dates = [b for b in right if _parse_date_text(b.text)]
    right_text = [b for b in right if not _parse_date_text(b.text)
                  and re.search(r"[A-Za-z]", b.text)
                  and not _is_junk_value(b.text)
                  and not _is_name(b)]

    # --- date of issue (left column, 2nd date after DOB) ---
    lbl_issue = _find_label(boxes, ["PENGELUARA", "DATEOFISS", "DATE OF ISS"])
    issue = (_value_below(boxes, lbl_issue, mrz_top, want_date=True)
             if lbl_issue else None)
    if not issue and len(left_dates) >= 2:
        issue = _parse_date_text(left_dates[1].text)  # 1st=DOB, 2nd=issue
    out["tanggal_terbit"] = issue

    # --- place of birth (right column, on the DOB row) ---
    excl = {name_n} if name_n else set()
    lbl_pob = _find_label(boxes, ["TEMPATLAHIR", "PLACEOFBIRTH", "PLACEOF"])
    pob_cands = (_values_below(boxes, lbl_pob, mrz_top, want_date=False,
                               exclude=excl) if lbl_pob else [])
    if right_text and left_dates:   # fallback: right-col text near the DOB row
        dob_y = left_dates[0].y
        near = sorted((b for b in right_text if b.y > mrz_top * 0.25),
                      key=lambda b: abs(b.y - dob_y))
        pob_cands += [b.text.strip().upper() for b in near]
    pob_val, pob_ok = _pick_place(pob_cands)
    out["tempat_lahir"] = pob_val
    out["_tempat_lahir_verified"] = pob_ok if pob_val else None

    # --- issuing office (right column, bottom text value) ---
    lbl_office = _find_label(boxes, ["MENGELUARKAN", "ISSUINGOFF",
                                     "ISSUINGOF", "KANTOR"])
    office_cands = (_values_below(boxes, lbl_office, mrz_top, want_date=False,
                                  exclude=excl) if lbl_office else [])
    # fallback: right-column text, bottom-first (office sits low on the page)
    office_cands += [b.text.strip().upper()
                     for b in sorted(right_text, key=lambda b: -b.y)]
    office_val, office_ok = _pick_place(office_cands)
    out["kantor_yang_mengeluarkan"] = office_val
    out["_kantor_verified"] = office_ok if office_val else None
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
FIELD_ORDER = [
    "jenis_type", "kode_negara", "no_paspor", "nama_lengkap",
    "kewarganegaraan", "tanggal_lahir", "sex", "tanggal_terbit",
    "tempat_lahir", "tanggal_habis_berlaku", "kantor_yang_mengeluarkan",
]


def _to_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d %b %Y")
    except ValueError:
        return None


def _dates_consistent(data: dict) -> bool:
    """Sanity-check the three dates: dob < issue < expiry, and a plausible
    validity span (Indonesian passports are 5 or 10 years). Catches OCR day/
    century slips that individual check digits can miss."""
    dob = _to_dt(data.get("tanggal_lahir"))
    iss = _to_dt(data.get("tanggal_terbit"))
    exp = _to_dt(data.get("tanggal_habis_berlaku"))
    if dob and iss and dob >= iss:
        return False
    if iss and exp:
        if exp <= iss:
            return False
        yrs = (exp - iss).days / 365.25
        if not (4 <= yrs <= 11):        # 5y or 10y booklets, with slack
            return False
    if dob and exp and dob >= exp:
        return False
    return True


def extract(src, engine: RapidOCR | None = None, name: str | None = None) -> dict:
    """Extract passport fields from a file path, raw image bytes, or an ndarray.
    `engine` lets a caller pass a pool-borrowed RapidOCR for thread-safe serving.
    `name` overrides the reported _file (useful when src is bytes)."""
    img = _load_image(src)
    boxes, m, angle = _best_orientation(img, engine)
    data: dict = {k: None for k in FIELD_ORDER}
    data["_file"] = name if name is not None else (
        Path(src).name if isinstance(src, (str, Path)) else None)
    data["_orientation"] = angle

    if m:
        for k in ("jenis_type", "kode_negara", "no_paspor", "nama_lengkap",
                  "kewarganegaraan", "tanggal_lahir", "sex",
                  "tanggal_habis_berlaku"):
            data[k] = m.get(k)
        data["_mrz_valid"] = m.get("_mrz_valid", {})
        data["_mrz_found"] = True
    else:
        data["_mrz_found"] = False

    # informational: the source scan carries a mobile-scanner watermark. Not a
    # data problem (extraction is unaffected) and NOT a review reason, but the
    # site can warn the user to re-scan before submitting to strict immigration.
    data["_scanner_watermark"] = any(_is_scanner_watermark(b.text)
                                     for b in boxes)

    mrz_top = _mrz_top(boxes)
    page_w = max((b.x1 for b in boxes), default=1000)
    vis = extract_visual(boxes, mrz_top, page_w, name=data.get("nama_lengkap") or "")
    for k, v in vis.items():
        data[k] = v

    # printed-name fallback when the MRZ name is missing or garbled (digits/too
    # short) — but only accept the printed value if it is cleaner
    if _name_looks_bad(data.get("nama_lengkap") or ""):
        printed = _printed_name(boxes, mrz_top)
        if printed and not _name_looks_bad(printed):
            data["nama_lengkap"] = printed

    # visual-zone fallback for the MRZ-only fields when the MRZ is unreadable,
    # so a clean scan is still fully extracted (values have no check digit, so
    # the record stays flagged for review via the "no_mrz" reason)
    if not data["_mrz_found"]:
        if not data.get("no_paspor"):
            data["no_paspor"] = _printed_passport_no(boxes, mrz_top)
        if not data.get("tanggal_lahir"):
            data["tanggal_lahir"] = _printed_dob(boxes, mrz_top)
        if not data.get("sex"):
            data["sex"] = _printed_sex(boxes, mrz_top)
        if data.get("no_paspor"):            # looks like a passport page
            data["jenis_type"] = data.get("jenis_type") or "P"
            data["kode_negara"] = data.get("kode_negara") or "IDN"
            data["kewarganegaraan"] = data.get("kewarganegaraan") or "IDN"

    # prefer unambiguous visual dates when MRZ missing / when present use MRZ
    # (kewarganegaraan: expand IDN -> INDONESIA for readability)
    if data.get("kewarganegaraan") == "IDN":
        data["kewarganegaraan"] = "INDONESIA"

    # actionable quality flag for the website: route these to manual review
    checks = data.get("_mrz_valid", {})
    reasons = []
    if not data["_mrz_found"]:
        reasons.append("no_mrz")
    if any(v is False for v in checks.values()):
        reasons.append("mrz_checkdigit_failed")
    if _name_looks_bad(data.get("nama_lengkap") or ""):
        reasons.append("name_unreliable")
    missing = [k for k in ("tanggal_terbit", "tempat_lahir",
                           "kantor_yang_mengeluarkan") if not data.get(k)]
    if missing:
        reasons.append("printed_fields_missing")
    if data.get("_tempat_lahir_verified") is False or \
            data.get("_kantor_verified") is False:
        reasons.append("place_unverified")
    if not _dates_consistent(data):
        reasons.append("date_inconsistent")
    # STRICT MODE: by default a scanner watermark is informational only. If you
    # need to reject watermarked scans (e.g. EU/UAE submissions), uncomment:
    # if data.get("_scanner_watermark"):
    #     reasons.append("scanner_watermark")
    data["_needs_review"] = bool(reasons)
    data["_review_reasons"] = reasons
    return data


def _extract_file(path: str) -> dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    try:
        return extract(path)
    except Exception as e:  # noqa
        return {"_file": Path(path).name, "_error": str(e)}


def _worker_init(threads: int):
    """Runs once per worker before any OCR. Caps each worker's OCR/OpenCV
    threads so workers x threads stays near the core count — avoiding the
    oversubscription that makes naive parallelism slower than serial."""
    os.environ["PPX_OCR_THREADS"] = str(threads)
    try:
        cv2.setNumThreads(threads)
    except Exception:  # noqa
        pass


def _run_batch(files, workers: int = 1) -> list[dict]:
    """Extract a list of files, optionally across worker processes. Processes
    (not threads) scale past the GIL and stay OS-agnostic. Each worker's OCR is
    given ~cores/workers threads so the machine isn't oversubscribed."""
    paths = [str(f) for f in files]
    if workers <= 1 or len(paths) <= 1:
        results = []
        for i, pth in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] {Path(pth).name}", file=sys.stderr)
            results.append(_extract_file(pth))
        return results
    per_worker = max(1, (os.cpu_count() or workers) // workers)
    print(f"Processing {len(paths)} files across {workers} workers "
          f"({per_worker} OCR threads each)…", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(per_worker,)) as ex:
        return list(ex.map(_extract_file, paths, chunksize=2))


def main():
    ap = argparse.ArgumentParser(description="Indonesian passport extractor")
    ap.add_argument("path", help="image file or folder")
    ap.add_argument("-o", "--output", help="output file (.json or .csv)")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into subfolders")
    ap.add_argument("--include-eds", action="store_true",
                    help="include (EDS)/(EDC) hand-annotated scans (skipped "
                         "by default; they carry signatures/stamps)")
    ap.add_argument("-j", "--workers", type=int, default=1,
                    help="parallel worker processes for folder batches "
                         "(default 1 = serial, which is fastest for typical "
                         "scans since CPU OCR is memory-bandwidth bound; only "
                         "raise it for large batches of heavy hi-res/rotated "
                         "scans, and benchmark first)")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        globber = p.rglob if args.recursive else p.glob
        files = sorted(f for f in globber("*") if f.suffix.lower() in exts)
        if not args.include_eds:
            files = [f for f in files
                     if not re.search(r"\((eds|edc)\)", f.stem, re.I)]
    else:
        files = [p]

    results = _run_batch(files, workers=max(1, args.workers))

    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".csv":
            cols = ["_file"] + FIELD_ORDER + ["_scanner_watermark",
                                              "_needs_review", "_review_reasons"]
            with out.open("w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in results:
                    row = {c: r.get(c, "") for c in cols}
                    if isinstance(row.get("_review_reasons"), list):
                        row["_review_reasons"] = ",".join(row["_review_reasons"])
                    w.writerow(row)
        else:
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"Wrote {len(results)} records -> {out}", file=sys.stderr)
    else:
        print(json.dumps(results if len(results) > 1 else results[0],
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
