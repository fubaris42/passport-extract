"""Analyse the raw-OCR dump for slip patterns that hurt anchoring/extraction.

Run:  uv run python analyze_ocr.py <ocr_dump.jsonl>
Reports:
  1. Label spellings   — how each field label actually gets OCR'd (drives the
     _find_label keywords and _is_label_text tokens).
  2. MRZ char slips     — raw vs check-digit-corrected line-2 fields, tallying
     which characters get confused (drives _CONFUSIONS / _fix_digits).
  3. Low-confidence text — boxes RapidOCR was unsure about.
  4. Sex / separator     — how the L/M · P/F field is rendered.
"""
import sys, json, re, difflib
from collections import Counter, defaultdict
import passport_extract as pe

recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
recs = [r for r in recs if "boxes" in r]
print(f"records: {len(recs)}\n")

norm = lambda t: re.sub(r"[^A-Z]", "", t.upper())

# canonical labels we care about (bilingual, as printed)
CANON = {
    "JENIS/TYPE": "JENISTYPE", "KODE NEGARA/COUNTRY CODE": "KODENEGARACOUNTRYCODE",
    "NO PASPOR/PASSPORT NO": "NOPASPORPASSPORTNO",
    "NAMA LENGKAP/FULL NAME": "NAMALENGKAPFULLNAME",
    "KEWARGANEGARAAN/NATIONALITY": "KEWARGANEGARAANNATIONALITY",
    "TGL LAHIR/DATE OF BIRTH": "TGLLAHIRDATEOFBIRTH",
    "KELAMIN/SEX": "KELAMINSEX",
    "TGL PENGELUARAN/DATE OF ISSUE": "TGLPENGELUARANDATEOFISSUE",
    "TEMPAT LAHIR/PLACE OF BIRTH": "TEMPATLAHIRPLACEOFBIRTH",
    "TGL HABIS BERLAKU/DATE OF EXPIRY": "TGLHABISBERLAKUDATEOFEXPIRY",
    "KANTOR/ISSUING OFFICE": "KANTORYANGMENGELUARKANISSUINGOFFICE",
    "NO REG": "NOREG",
}

# ---- 1. label spellings ----------------------------------------------------
variants = defaultdict(Counter)
for r in recs:
    for b in r["boxes"]:
        n = norm(b[0])
        if len(n) < 4:
            continue
        best, bestr = None, 0.0
        for canon, ck in CANON.items():
            ratio = difflib.SequenceMatcher(None, n, ck).ratio()
            if ratio > bestr:
                best, bestr = canon, ratio
        if bestr >= 0.6:               # clearly a mangled label
            variants[best][b[0].strip()] += 1

print("=" * 70)
print("1. LABEL SPELLINGS (canonical -> observed OCR variants x count)")
print("=" * 70)
for canon in CANON:
    vs = variants.get(canon)
    if not vs:
        continue
    print(f"\n{canon}")
    for txt, c in vs.most_common(8):
        print(f"   {c:3}  {txt!r}")

# ---- 2. MRZ character slips (raw vs check-digit-corrected) ------------------
slips = Counter()
field_fixed = Counter()
for r in recs:
    l2 = r.get("mrz_l2")
    if not l2 or len(l2) < 28:
        continue
    for name, raw, chk in (("doc", l2[0:9], l2[9]),
                           ("dob", l2[13:19], l2[19]),
                           ("exp", l2[21:27], l2[27])):
        fixed, ok = pe._correct_field(raw, chk)
        if ok and fixed != raw:
            field_fixed[name] += 1
            for a, b in zip(raw, fixed):
                if a != b:
                    slips[f"{a}->{b}"] += 1

print("\n" + "=" * 70)
print("2. MRZ CHAR SLIPS (from check-digit corrections that changed a field)")
print("=" * 70)
print("fields repaired:", dict(field_fixed))
for k, c in slips.most_common(20):
    print(f"   {c:3}  {k}")

# ---- 3. low-confidence boxes ----------------------------------------------
low = Counter()
for r in recs:
    for b in r["boxes"]:
        if b[1] < 0.6 and len(b[0].strip()) >= 2:
            low[b[0].strip()] += 1
print("\n" + "=" * 70)
print("3. LOW-CONFIDENCE BOXES (score < 0.6)")
print("=" * 70)
for txt, c in low.most_common(25):
    print(f"   {c:3}  {txt!r}")

# ---- 4. sex / separator rendering -----------------------------------------
sexes = Counter()
for r in recs:
    for b in r["boxes"]:
        u = b[0].upper().replace(" ", "")
        if re.fullmatch(r"[LMPF][^A-Za-z0-9]{0,2}[LMPF]?", u) and len(u) <= 4:
            sexes[b[0].strip()] += 1
print("\n" + "=" * 70)
print("4. SEX FIELD RENDERINGS")
print("=" * 70)
for txt, c in sexes.most_common(15):
    print(f"   {c:3}  {txt!r}")
