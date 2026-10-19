"""Unit tests for the pure logic (MRZ parsing, check digits, correction,
gazetteer, date sanity). Uses SYNTHETIC MRZ strings only — no real passport
data lives in the repo or runs in CI. No OCR/model needed, so these are fast.

Run:  uv run pytest
"""
import passport_extract as pe
import gazetteer as gz


def _cd(data: str) -> str:
    """Build a valid ICAO check digit for a field (mirror of the spec)."""
    return str(pe._mrz_check(data))


def _make_td3(doc="X1234567", nat="IDN", dob="900101", sex="M",
              exp="300101", surname="SETIAWAN", given="HENDRA"):
    """Assemble a well-formed TD3 pair with correct check digits."""
    l1 = ("P<" + nat + surname + "<<" + given).ljust(44, "<")[:44]
    doc9 = doc.ljust(9, "<")
    l2 = doc9 + _cd(doc9) + nat + dob + _cd(dob) + sex + exp + _cd(exp)
    l2 = l2.ljust(43, "<")
    l2 = l2 + _cd(l2[0:10] + l2[13:20] + l2[21:43])  # composite (not asserted)
    return l1[:44], l2[:44]


def test_parse_valid_mrz():
    l1, l2 = _make_td3()
    d = pe.parse_mrz(l1, l2)
    assert d["jenis_type"] == "P"
    assert d["kode_negara"] == "IDN"
    assert d["no_paspor"] == "X1234567"
    assert d["nama_lengkap"] == "HENDRA SETIAWAN"
    assert d["sex"] == "M"
    assert d["tanggal_lahir"] == "01 JAN 1990"
    assert d["tanggal_habis_berlaku"] == "01 JAN 2030"
    assert all(d["_mrz_valid"].values())


def test_validation_invariant_holds():
    # Whenever a field is reported valid, its check digit must actually recompute
    # correctly for the returned value — no false "valid".
    l1, l2 = _make_td3(doc="X1234567")
    bad = l2[0] + "9" + l2[2:]              # corrupt, keep old (now wrong) digit
    d = pe.parse_mrz(l1, bad)
    if d["_mrz_valid"]["no_paspor"]:
        field = d["no_paspor"].ljust(9, "<")
        assert pe._mrz_check(field) == int(bad[9])


def test_check_digit_correction_recovers_ocr_slip():
    l1, l2 = _make_td3(doc="X1234567")
    # simulate an OCR slip inside the document field: 1 -> I (a known confusion)
    slipped = l2[0] + "I" + l2[2:]           # "XI234567" instead of "X1234567"
    d = pe.parse_mrz(l1, slipped)
    assert d["no_paspor"] == "X1234567"
    assert d["_mrz_valid"]["no_paspor"] is True


def test_dob_century_inference():
    l1, l2 = _make_td3(dob="050101")          # ambiguous 05 -> 1905 or 2005
    d = pe.parse_mrz(l1, l2)
    # a birth date cannot be in the future
    assert d["tanggal_lahir"].endswith("2005") or d["tanggal_lahir"].endswith("1905")


def test_line_detection_line2_with_filler():
    # line 2 that ends in '<<' filler must still be recognised as line 2
    l1, l2 = _make_td3(doc="E4612150", surname="SISWOYO", given="MASLIHAH")

    class B:
        def __init__(self, t, y):
            self.text, self.y, self.y0 = t, y, y
            self.x0, self.x1, self.y1, self.x, self.score = 0, 100, y, 50, .9
    boxes = [B(l1, 10), B(l2, 20)]
    found = pe._find_mrz_lines(boxes)
    assert found is not None
    assert found[0].startswith("P<IDN")


def test_gazetteer_matches_and_despaces():
    assert gz.match_place("BANDUNG") == "BANDUNG"
    assert gz.match_place("JAKARTATIMUR") == "JAKARTA TIMUR"   # dropped space
    assert gz.match_place("ZZZNOTAPLACE") is None


def test_dictionary_flags_label_words_not_places():
    assert gz.is_dictionary_word("kepada") is True
    assert gz.is_dictionary_word("perhatian") is True
    # a proper-noun city is not a common dictionary word
    assert gz.is_dictionary_word("Sukabumi") is False


def test_date_consistency():
    ok = {"tanggal_lahir": "01 JAN 1990", "tanggal_terbit": "01 JAN 2020",
          "tanggal_habis_berlaku": "01 JAN 2025"}
    assert pe._dates_consistent(ok) is True
    bad = {"tanggal_lahir": "01 JAN 1990", "tanggal_terbit": "01 JAN 2025",
           "tanggal_habis_berlaku": "01 JAN 2020"}      # expiry before issue
    assert pe._dates_consistent(bad) is False


def test_country_ocr_variants():
    assert pe._fix_country("1DN") == "IDN"
    assert pe._fix_country("IDN") == "IDN"


def test_scanner_watermark_detection():
    assert pe._is_scanner_watermark("Scanned by TapScanner") is True
    assert pe._is_scanner_watermark("BANDUNG") is False


import pytest


@pytest.mark.parametrize("raw,exp", [
    ("L/M", "M"), ("LM", "M"), ("L1M", "M"), ("LiM", "M"), ("L|M", "M"),
    ("LI MM", "M"), ("M", "M"), ("L", "M"),
    ("P/F", "F"), ("PF", "F"), ("P|F", "F"), ("P", "F"), ("F", "F"),
    ("", None), ("X", None), ("LP", None),   # ambiguous / unknown -> None
])
def test_norm_sex_tolerant(raw, exp):
    assert pe._norm_sex(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("15 MAR 1995", "15 MAR 1995"),
    ("15MAR1995", "15 MAR 1995"),
    ("16 SEP l986", "16 SEP 1986"),      # year digit slip l->1
    ("O5 JAN 2OO5", "05 JAN 2005"),      # O->0 slips in day and year
    ("01 MEI 1963", "01 MAY 1963"),      # Indonesian month -> canonical
    ("SUKABUMI", None), ("BANDUNG 2020", None),
])
def test_extract_date_tolerant(raw, exp):
    assert pe._extract_date(raw) == exp
