"""Dump raw RapidOCR output for every scan in a folder, for slip analysis.

Writes one JSON object per image to a .jsonl file:
  {file, angle, mrz_found, mrz_l1, mrz_l2, boxes:[[text,score,x0,y0,x1,y1],...]}

Boxes are from the chosen upright/visual orientation (what anchoring sees).
Run:  uv run python dump_ocr.py <folder> <out.jsonl>
"""
import sys, json, re
from pathlib import Path
import passport_extract as pe

folder, out = Path(sys.argv[1]), Path(sys.argv[2])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
files = sorted(f for f in folder.rglob("*")
               if f.suffix.lower() in exts
               and not re.search(r"\((eds|edc)\)", f.stem, re.I))

with out.open("w", encoding="utf-8") as fh:
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr)
        try:
            img = pe._load_image(str(f))
            boxes, mrz, angle = pe._best_orientation(img)
            lines = pe._find_mrz_lines(boxes)
            rec = {
                "file": f.name,
                "angle": angle,
                "mrz_found": mrz is not None,
                "mrz_l1": lines[0] if lines else None,
                "mrz_l2": lines[1] if lines else None,
                "boxes": [[b.text, round(b.score, 2),
                           int(b.x0), int(b.y0), int(b.x1), int(b.y1)]
                          for b in boxes],
            }
        except Exception as e:  # noqa
            rec = {"file": f.name, "_error": str(e)}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {len(files)} records -> {out}", file=sys.stderr)
