"""
Step 1 of DocSight: PDF ingestion.

Turns a folder of PDFs into two things every later step depends on:

  1. A rendered PNG image for every page (so charts/tables/diagrams stay
     intact — we are NOT throwing that away by extracting text only).
  2. A metadata.jsonl file — one JSON record per page — that ties the
     image back to its raw extracted text, source document, and page number.

📚 Concept: the page IMAGE is a first-class artifact here, not a byproduct.
Step 2 (multimodal indexing) will embed these images directly (ColPali) or
alongside CLIP + text embeddings — that's the whole "visual RAG" idea, and
it starts here: never let the chart get OCR'd away into lossy text.

Usage:
    python -m src.ingest --input data/pdfs --output output/
"""

import argparse
import json
from pathlib import Path

import pymupdf  # PyMuPDF — opens PDFs and rasterizes pages to images ("fitz" is its older, deprecated import name)
from tqdm import tqdm  # progress bar — purely cosmetic, nice for large PDF sets


def render_page_image(page: pymupdf.Page, out_path: Path, dpi: int = 150) -> None:
    """
    Render one PDF page to a PNG file.

    dpi=150 is a deliberate tradeoff: high enough that small chart labels and
    table text stay legible to a vision-language model in Step 4, low enough
    that a 40-page PDF doesn't produce huge images that blow your API
    image-token budget or GPU memory in Step 2.

    ⚠️ Common mistake: cranking dpi to 300+ "for quality." Most VLMs
    downsample large images internally anyway, so you pay in tokens/memory
    for resolution the model never actually uses.
    """
    pix = page.get_pixmap(dpi=dpi)  # Rasterizes the page — this is what "sees" the chart
    pix.save(str(out_path))


def ingest_pdf(pdf_path: Path, images_dir: Path, dpi: int = 150) -> list[dict]:
    """
    Ingest a single PDF: render every page to an image and extract its text.

    Returns a list of one metadata dict per page — this dict is the unit
    Step 2 will embed and Step 3 will retrieve, so its shape matters: keep
    it flat and JSON-serializable, nothing fancier is needed yet.
    """
    doc_id = pdf_path.stem  # filename without ".pdf" — a human-readable id you'll see in citations later
    records = []

    with pymupdf.open(pdf_path) as doc:  # `with` guarantees the file handle closes even on error
        for page_num, page in enumerate(doc):
            image_path = images_dir / f"{doc_id}_p{page_num:03d}.png"
            render_page_image(page, image_path, dpi=dpi)

            text = page.get_text().strip()  # Raw text layer — empty for scanned/image-only pages

            records.append({
                "doc_id": doc_id,
                "page_num": page_num,
                "source_pdf": str(pdf_path.name),
                "image_path": str(image_path),
                "text": text,
                "char_count": len(text),
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into page images + text metadata.")
    parser.add_argument("--input", type=Path, default=Path("data/pdfs"), help="Folder of PDFs to ingest")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Where to write pages/ and metadata.jsonl")
    parser.add_argument("--dpi", type=int, default=150, help="Render resolution — see render_page_image docstring")
    args = parser.parse_args()

    images_dir = args.output / "pages"
    images_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(args.input.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {args.input}/ — drop some in there first.")
        return

    all_records = []
    for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
        all_records.extend(ingest_pdf(pdf_path, images_dir, dpi=args.dpi))

    metadata_path = args.output / "metadata.jsonl"
    # JSON Lines (one JSON object per line) rather than one big JSON array:
    # it streams — you can append, tail -f it while it runs, or load it lazily
    # line-by-line in Step 2 without holding the whole file in memory.
    with open(metadata_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # --- Sanity-check summary — always read these numbers before moving on ---
    # 💡 Pro tip: this kind of "did the pipeline do something sane" printout,
    # not just "it didn't crash," is exactly the habit the evaluation step
    # later in the guide builds on. Start it here.
    num_docs = len(pdf_paths)
    num_pages = len(all_records)
    empty_text_pages = sum(1 for r in all_records if r["char_count"] == 0)
    avg_chars = sum(r["char_count"] for r in all_records) / max(num_pages, 1)

    print(f"\nIngested {num_docs} PDFs -> {num_pages} pages")
    print(f"Images:   {images_dir}/")
    print(f"Metadata: {metadata_path}")
    print(f"Avg text per page: {avg_chars:.0f} chars")
    if empty_text_pages:
        print(f"⚠️  {empty_text_pages} pages had NO extracted text (likely scanned/image-only pages).")
        print("    That's fine here — Step 2's page-image embeddings don't need OCR text at all.")
        print("    If you also want text search over those pages, you'd need OCR (e.g. pytesseract).")


if __name__ == "__main__":
    main()
