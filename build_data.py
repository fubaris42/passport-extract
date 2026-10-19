"""One-off: build data/ files for the gazetteer from source lists.

Sources (not shipped, only needed to regenerate):
  - indonesian_reg.txt          514 kabupaten/kota + provinces
  - indonesian-wordlist/00-indonesian-wordlist.lst  ~80k Indonesian words

Outputs (shipped, loaded at runtime):
  - data/id_places.txt   canonical UPPER place names (birthplace + office)
  - data/id_words.txt    lowercase Indonesian dictionary words (junk filter)
  - data/id_offices_extra.txt  overseas missions / forms not in the region list
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# --- places from the official region list -------------------------------
places = set()
for line in (HERE / "indonesian_reg.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) < 2:
        continue
    kind, name = parts[0], parts[1]
    name = re.sub(r"\(.*?\)", "", name)          # drop "(SUMUT)"
    name = re.sub(r"^Administrasi\s+", "", name, flags=re.I)  # DKI kota
    name = name.strip()
    if kind.lower() == "provinsi":
        # drop leading province codes like "NAD", "DKI"
        toks = name.split()
        if toks and toks[0].isupper() and len(toks[0]) <= 4 and len(toks) > 1:
            toks = toks[1:]
        name = " ".join(toks)
    if name:
        places.add(name.upper())

# common bare forms + Kanim office towns the kab/kota list lacks but that
# passports print as birthplace / issuing office
extras = [
    "JAKARTA", "YOGYAKARTA", "BATAM", "BATUSANGKAR", "TAKENGON",
    "RANTAUPRAPAT", "SOEKARNO HATTA", "BUKITTINGGI",
]
places.update(extras)

(DATA / "id_places.txt").write_text(
    "\n".join(sorted(places)) + "\n", encoding="utf-8")
print(f"id_places.txt: {len(places)} places")

# --- overseas issuing offices (KBRI/KJRI missions) ----------------------
# format: the printed office text; matched as-is (plus a KBRI/KJRI prefix rule
# in code catches any city we didn't list)
offices_extra = [
    "LUAR NEGERI", "SOEKARNO HATTA",
    "KBRI SINGAPURA", "KBRI KUALA LUMPUR", "KBRI CAIRO", "KBRI RIYADH",
    "KBRI ABU DHABI", "KBRI TOKYO", "KBRI SEOUL", "KBRI BEIJING",
    "KJRI JEDDAH", "KJRI SYDNEY", "KJRI HONG KONG", "KJRI DUBAI",
    "KJRI PENANG", "KJRI JOHOR BAHRU", "KJRI KUCHING", "KJRI KOTA KINABALU",
]
(DATA / "id_offices_extra.txt").write_text(
    "\n".join(offices_extra) + "\n", encoding="utf-8")
print(f"id_offices_extra.txt: {len(offices_extra)} offices")

# --- Indonesian dictionary (junk / label filter) ------------------------
src = HERE / "indonesian-wordlist" / "00-indonesian-wordlist.lst"
words = set()
for w in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    w = w.strip().lower()
    if w.isalpha() and len(w) >= 3:      # skip 1-2 char noise, hyphenates
        words.add(w)
(DATA / "id_words.txt").write_text(
    "\n".join(sorted(words)) + "\n", encoding="utf-8")
print(f"id_words.txt: {len(words)} words")
