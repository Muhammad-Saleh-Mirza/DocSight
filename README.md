# DocSight

Multimodal visual RAG over PDFs — ask questions, get answers grounded in the
actual page image (charts, tables, diagrams included), not just OCR'd text.

Full step-by-step plan / architecture / resume notes: see the project guide
(https://claude.ai/artifact/FQ5QN51jjUh4KM1upnpmA3) — this repo is the code
that follows it, built one step at a time.

## Status

- [x] Step 1 — Ingestion (this commit)
- [ ] Step 2 — Multimodal indexing
- [ ] Step 3 — Retrieval + reranking
- [ ] Step 4 — Grounded generation
- [ ] Step 5 — Evaluation
- [ ] Step 6 — Demo UI
- [ ] Step 7 — Deployment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 — Ingest PDFs

Drop a few PDFs into `data/pdfs/`, then:

```bash
python -m src.ingest --input data/pdfs --output output/
```

This writes:
- `output/pages/*.png` — one rendered image per page (150 dpi)
- `output/metadata.jsonl` — one JSON record per page: `doc_id`, `page_num`,
  `source_pdf`, `image_path`, `text`, `char_count`

The console summary at the end tells you page count, average text length, and
how many pages had no extractable text (scanned/image-only pages — expected,
not a bug).

Read `src/ingest.py` — every non-trivial line is commented, including *why*
each design choice was made (e.g. why 150 dpi, why we keep the image at all).
