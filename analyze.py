"""Summarise extraction quality from a CSV produced by passport_extract.py."""
import csv, sys, re
from collections import Counter

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
n = len(rows)
FIELDS = ["jenis_type","kode_negara","no_paspor","nama_lengkap","kewarganegaraan",
          "tanggal_lahir","sex","tanggal_terbit","tempat_lahir",
          "tanggal_habis_berlaku","kantor_yang_mengeluarkan"]

print(f"total rows: {n}\n")
print("empty-field counts:")
for f in FIELDS:
    empty = sum(1 for r in rows if not (r.get(f) or "").strip())
    print(f"  {f:26} {empty:3}  ({100*empty/n:.0f}%)")

no_core = [r for r in rows if not (r.get("no_paspor") or "").strip()]
print(f"\nno passport no (likely no MRZ / bad page): {len(no_core)}")
for r in no_core:
    print("   -", r["_file"])

DATE = re.compile(r"^\d{2} [A-Z]{3} \d{4}$")
print("\nsuspicious values:")
for r in rows:
    f = r["_file"]; issues = []
    nm = (r.get("nama_lengkap") or "")
    if re.search(r"\b[A-Z]\b", nm): issues.append(f"name-stray-letter:{nm!r}")
    if re.search(r"\d", nm): issues.append(f"name-digit:{nm!r}")
    for df in ("tanggal_lahir","tanggal_terbit","tanggal_habis_berlaku"):
        v=(r.get(df) or "").strip()
        if v and not DATE.match(v): issues.append(f"{df}-badfmt:{v!r}")
    sx=(r.get("sex") or "").strip()
    if sx and sx not in ("M","F"): issues.append(f"sex:{sx!r}")
    kn=(r.get("kode_negara") or "").strip()
    if kn and kn!="IDN": issues.append(f"country:{kn!r}")
    for pf in ("tempat_lahir","kantor_yang_mengeluarkan"):
        v=(r.get(pf) or "").strip()
        if v and (any(c.isdigit() for c in v) or "/" in v): issues.append(f"{pf}:{v!r}")
    if issues:
        print(f"   {f}: " + "; ".join(issues))

print("\nissuing office value distribution:")
for v,c in Counter((r.get("kantor_yang_mengeluarkan") or "").strip() for r in rows).most_common():
    print(f"  {c:3}  {v!r}")
