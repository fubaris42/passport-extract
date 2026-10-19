# Passport Data Extractor (Indonesian passports)

Extracts the 11 requested fields from Indonesian passport photos, for use as a
website feature. Reads the **MRZ** (machine-readable zone, check-digit verified)
for the reliable fields and the **printed zone** for the three fields the MRZ
does not contain.

## Fields

| Output key | Label on passport | Source |
|---|---|---|
| `jenis_type` | Jenis / Type | MRZ |
| `kode_negara` | Kode Negara / Country Code | MRZ |
| `no_paspor` | No. Paspor / Passport No. | MRZ (✓ check digit) |
| `nama_lengkap` | Nama Lengkap / Full Name | MRZ |
| `kewarganegaraan` | Kewarganegaraan / Nationality | MRZ |
| `tanggal_lahir` | Tgl Lahir / Date of Birth | MRZ (✓ check digit) |
| `sex` | Kelamin / Sex | MRZ |
| `tanggal_terbit` | Tgl Pengeluaran / Date of Issue | printed zone |
| `tempat_lahir` | Tempat Lahir / Place of Birth | printed zone |
| `tanggal_habis_berlaku` | Tgl Habis Berlaku / Date of Expiry | MRZ (✓ check digit) |
| `kantor_yang_mengeluarkan` | Kantor yang Mengeluarkan / Issuing Office | printed zone |

The MRZ fields are highly reliable (validated by ICAO check digits — see
`_mrz_valid` in the output). The three printed-zone fields are best-effort OCR
and should be treated as lower confidence, especially on low-quality or
heavily-stamped scans.

## Robustness to scan quality & layout

- **Orientation** is detected by text-box *shape* (wide = horizontal lines),
  not by reading words — RapidOCR's angle classifier reads content at any
  rotation, so word-based detection can't tell upright from rotated. Visual
  fields and the MRZ are oriented **independently**: RapidOCR sometimes drops an
  MRZ line when the page is upright but reads it when rotated, so we take the
  printed fields from the upright orientation and the MRZ from whichever
  orientation validates.
- **Resolution-independent**: works from ~700px to 3500px+ scans.
- **MRZ line detection** tells the two lines apart by digit content (line 2 is
  digit-heavy and may contain `<<` filler runs), not by `<<` presence.
- **Visual-zone fallback**: when the MRZ is unreadable (e.g. security-pattern
  background defeats OCR), the core fields (passport no, DOB, sex, name) are
  read from the printed labels instead — so a clean scan is still fully
  extracted. These have no check digit, so the record stays flagged `no_mrz`
  for a human glance.

## How it handles messy real-world scans

- **Rotation**: scans rotated 90/180/270° are auto-corrected — it OCRs each
  orientation and keeps the one with a check-digit-valid MRZ (`_orientation` in
  the output shows the angle used).
- **Unicode paths**: reads files with non-ASCII names on Windows.
- **Hand-annotated pages**: files whose name ends in `(EDS)` / `(EDC)` carry
  signatures, stamps and handwriting and are **skipped by default** in batch
  mode (pass `--include-eds` to force them).
- **Non-bio pages / unreadable scans**: if no MRZ is found, `_mrz_found` is
  `false` — the record should be routed to manual review.

## Install

```bash
uv sync
```

## CLI

```bash
# single image -> JSON on stdout
uv run python passport_extract.py "path/to/passport.jpg"

# a whole folder -> CSV (recurse into subfolders, skip EDS files)
uv run python passport_extract.py "pass/" --recursive -o out.csv

# JSON array output
uv run python passport_extract.py "pass/" --recursive -o out.json
```

## Web service (production-grade, OS-agnostic)

```bash
uv run python web.py          # serves via waitress on 0.0.0.0:5000
```

- `GET  /`            — drag-and-drop upload page.
- `GET  /health`      — liveness/readiness probe (no PII).
- `POST /api/extract` — multipart form; field `image` (single) or `images`
  (multiple). Returns a JSON object (single) or array (multiple).

```bash
curl -F "image=@passport.jpg" http://localhost:5000/api/extract
```

Production characteristics:

- **No PII on disk** — uploads are decoded in memory (`cv2.imdecode`), never
  written to a temp file.
- **Cross-platform WSGI** — served by **waitress** (Windows + Linux), not the
  Flask dev server. Model is warmed at startup.
- **Thread-safe concurrency** — a pool of OCR engines (`EnginePool`) is borrowed
  per request, so concurrent uploads never share a non-thread-safe RapidOCR
  instance. No process forking, so it behaves identically on every OS.
- **PII-safe logging** — structured JSON with counts/latency/review only; field
  values are never logged.

Config via environment:

| Var | Default | Meaning |
|---|---|---|
| `PPX_POOL_SIZE` | 2 | OCR engines in the pool (concurrency) |
| `PPX_THREADS` | 4×pool | waitress worker threads |
| `PPX_PORT` | 5000 | listen port |
| `PPX_MAX_UPLOAD_MB` | 20 | max request size |
| `PPX_MAX_FILES` | 10 | max images per request |
| `PPX_MAX_EDGE` | 1800 | image downscale long-edge |

**Concurrency tuning:** RapidOCR/ONNXRuntime already parallelises each inference
across cores, so a large pool on top oversubscribes the CPU. Keep `PPX_POOL_SIZE`
small (2–4). For heavy concurrent load, raise the pool and set
`OMP_NUM_THREADS=1` to make each engine single-threaded (better tail latency).

## Batch CLI

```bash
uv run python passport_extract.py "pass/" --recursive -o out.csv
```

**Serial (default) is the right choice for most batches.** CPU OCR here is
memory-bandwidth bound, so on typical scans one process using all cores beats
splitting into workers (measured: 42 files → 2m0s serial vs 3m36s at `-j 4`).

`-j/--workers` (ProcessPoolExecutor, cross-platform) only pays off for **large
batches of heavy hi-res/rotated scans** that each do multiple OCR passes — there
it gave a ~4× speedup. Benchmark on your own data before raising it; each worker
is capped to ~cores/workers threads to avoid CPU oversubscription.

## Tests

```bash
uv run pytest
```

Unit tests cover MRZ parsing, check-digit validation **and correction**,
gazetteer matching, date sanity and watermark detection — all on **synthetic**
MRZ strings. No real passport data is in the repo or CI, and no OCR model is
needed, so the suite is fast.

## Scanner watermarks

Mobile scanner apps (TapScanner, CamScanner, Adobe Scan…) stamp a watermark at
the image bottom. It is **not** part of the passport, so it is filtered out of
all fields and never triggers review. Instead each record carries an
informational `_scanner_watermark` boolean — use it to warn the user to re-scan
without the watermark before submitting to strict immigration (EU, UAE), while
still accepting the extracted data. The app-name set is a small fixed list in
`_SCANNER_TOKENS`.

## MRZ check-digit correction & date sanity

- The MRZ carries ICAO check digits. When a field's digit fails, the extractor
  tries the plausible OCR confusions (`O/0, I/1, S/5, B/8, …`, single then
  double) until the check digit passes — recovering the true value instead of
  just flagging it. A correction is only accepted if the check digit validates.
- Cross-field **date sanity**: `dob < issue < expiry` and a plausible 5/10-year
  validity span. Violations raise `date_inconsistent` for review — catching
  day/century OCR slips that a single field's check digit can miss.

## Review flag (route bad scans automatically)

Every record carries `_needs_review` (bool) and `_review_reasons` (list). Reasons:

- `no_mrz` — MRZ not detected (wrong page / unreadable scan)
- `mrz_checkdigit_failed` — a check digit didn't validate (suspect OCR)
- `name_unreliable` — name has digits or is implausibly short
- `printed_fields_missing` — one of the 3 printed-only fields wasn't found
- `place_unverified` — birthplace/office didn't match the gazetteer
- `date_inconsistent` — the dates fail the dob<issue<expiry / validity-span check

Use it to auto-accept clean records and send only the flagged ones to a human.
These columns are included in CSV output too.

## Accuracy on the 221-image sample set

- MRZ detected + all check digits valid: **~210/221 (95%)**
- Names clean: **~217/221 (98%)** — the rest are digit-garbled MRZ on poor
  scans (a recognition limit, all flagged `name_unreliable`)
- Issuing office / place of birth: real place names, incl. overseas offices
  (KBRI/KJRI); reg-number and label-fragment leaks eliminated
- The 11 "no passport no" cases are non-bio pages, child passports with an
  odd layout, or scans too poor to read — all flagged for review

## Data-driven place validation (no hardcoded blacklists)

Place fields are validated against static data instead of hand-maintained
blacklists, so the logic stays small and the knowledge lives in editable files:

- `data/id_places.txt` — 523 Indonesian provinces + kabupaten/kota + a few
  Kanim office towns (for `tempat_lahir` and `kantor`)
- `data/id_offices_extra.txt` — overseas missions (KBRI/KJRI …)
- `data/id_words.txt` — ~78k Indonesian dictionary words

[gazetteer.py](gazetteer.py) does the matching (stdlib `difflib`, no new deps):

- **`match_place`** snaps an OCR value to a real place — exact, then despaced
  (`JAKARTATIMUR` → `JAKARTA TIMUR`), then fuzzy for small OCR slips. This one
  function replaced the old spacing-fix dict *and* rejects reg-numbers/label
  fragments for free (they match no place).
- **`is_dictionary_word`** flags label/notes words (`kepada`, `perhatian`,
  `paspor`) — a token that is a common word and *not* a place is not a value.
  This replaced the growing `_JUNK_TOKENS` list.

Unmatched but plausible values are **kept raw and flagged** `place_unverified`
(in `_review_reasons`) rather than dropped — so gaps in the gazetteer degrade
gracefully. To extend coverage, just edit the data files (or re-run
`build_data.py` against fuller source lists); no code change needed.

## Design decisions (kept intentionally lean)

- **CPU-only stack**: RapidOCR (ONNX) + OpenCV. No Tesseract binary, no GPU, no
  transformer models — it runs on a modest VM without special specs.
- **No LayoutLM / document-AI model**: passports are a *fixed* ISO TD3 layout, so
  there is no variable layout to learn. Our residual errors are character
  *recognition* on poor scans, which a layout model (it consumes OCR tokens, it
  doesn't produce them) cannot fix. It would add heavy deps for no gain here.
- **No VLM fallback by default**: too heavy per-request for a web feature. If you
  ever want maximum recovery on the flagged ~5%, gate a VLM behind
  `_needs_review` so it only runs on the few bad scans — never on every upload.

## Notes

- Handles real, personal passport data — keep uploads and any stored output
  secured, and don't log full records in plaintext in production.
- Fields not present in the MRZ have no check digit, so validate
  `tanggal_terbit`, `tempat_lahir` and `kantor_yang_mengeluarkan` before
  trusting them downstream.
