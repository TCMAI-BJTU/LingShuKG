# LingShuKG

This repository provides the literature-based knowledge extraction pipeline used in the construction of LingShu, a large-scale symptom-centric contextualized knowledge graph integrating Traditional Chinese Medicine (TCM) and modern biomedicine.

The LingShu knowledge graph can be explored through the online platform: **http://www.tcmkg.com/**

The pipeline converts TCM literature from PDF to text using `kg_ocr` and extracts ontology-constrained entities, triples, and contextualized quadruples using the `kg_agent` extraction agent.

## Pipeline

```text
PDF (optional)
  → kg_ocr (OCR → TXT)
  → data/ (or kg_ocr/output/)
  → kg_agent / main.py (knowledge-graph extraction)
  → output/ + state/kg.sqlite
```

Sample text is ready at `data/冠心病全国名老中医治验集萃.txt` and can be used for extraction immediately.

## Layout

| Path | Description |
|------|-------------|
| `data/` | Text data (current sample: `冠心病全国名老中医治验集萃.txt`) |
| `kg_ocr/` | PDF → TXT OCR pipeline |
| `kg_ocr/output/` | OCR output directory (for batch PDF recognition) |
| `kg_agent/` | Knowledge-graph extraction agent |
| `schemas/example.json` | Entity / relation schema |
| `main.py` / `run.sh` | Directory-level extraction entry points |
| `state/` | SQLite graph store and checkpoint cache |
| `output/` | Extraction result JSON (mirrors source tree) |
| `sqlite2csv.py` | Export `state/kg.sqlite` to CSV |

## Setup

1. Activate the matching Conda environment on your machine.
2. Start the OpenAI-compatible LLM service used by OCR / extraction (default `http://127.0.0.1:8000/v1`).
3. Place PDFs under project-root `data/` (or any directory, and adjust the command paths).

LLM settings can be overridden with env vars such as `KG_LLM_BASE_URL`, `KG_LLM_API_KEY`, and `KG_LLM_MODEL`.

## Step 1: PDF → TXT (`kg_ocr`)

Run under `kg_ocr/` (or pass the script path from the project root). Output defaults to `kg_ocr/output/` as `txt`.

```bash
cd kg_ocr

# Single PDF
python ocr_qwen.py /path/to/file.pdf --format txt

# Recursively process a directory
python ocr_qwen.py ../data --recursive --workers 128 --format txt
```

Optional two-stage flow: extract high-confidence native PDFs first, then run visual OCR on the rest:

```bash
python ocr_qwen.py ../data --recursive --native-only --file-workers 32 --format txt
python ocr_qwen.py ../data --recursive --workers 128 --format txt
```

See `kg_ocr/README.md` for the DeepSeek-OCR-2 multi-port variant.

Existing complete TXT files are skipped by default; add `--overwrite` to reprocess.

## Step 2: TXT → Knowledge Graph (`kg_agent`)

Run from the project root. Input should be a directory of TXT files (from OCR or prepared text):

```bash
# Recommended wrapper (reads data/)
bash run.sh

# Or call directly
python main.py \
  --data-dir data \
  --output-dir output \
  --schema schemas/example.json
```

Common arguments:

| Argument | Description |
|----------|-------------|
| `--data-dir` | TXT / Markdown input directory |
| `--output-dir` | JSON result output directory |
| `--schema` | Schema JSON; default `schemas/example.json` |
| `--workers` | Concurrent chunks; default `128` |
| `--debug` | Stream traces and force serial execution |

Results are written to `output/`; graph state is stored in `state/kg.sqlite` with resume support.

## Export CSV (optional)

```bash
python sqlite2csv.py
```

Exports `state/kg.sqlite` to `csv_output/` by default (`sources.csv` / `entities.csv` / `relations.csv`).

## More detail

OCR module details, concurrency settings, and native-extraction strategy: [`kg_ocr/README.md`](kg_ocr/README.md).

## Data Availability

The LingShu web platform is available for online exploration. The complete dataset associated with the manuscript will be made publicly available for download upon acceptance.

The LingShu knowledge graph can be explored through the online platform: **http://www.tcmkg.com/**
