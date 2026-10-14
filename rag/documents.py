from pathlib import Path

import fitz

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_pages(pdf_path: Path) -> list[dict]:
    with fitz.open(pdf_path) as doc:
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    return [{"page": num, "text": text} for num, text in pages if text.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current += (" " if current else "") + para
            continue

        if current:
            chunks.append(current)
        if len(para) > chunk_size:
            step = chunk_size - overlap
            chunks.extend(para[i:i + chunk_size] for i in range(0, len(para), step))
            current = ""
        else:
            current = para

    if current:
        chunks.append(current)

    # Prepend the tail of the previous chunk so context isn't lost at boundaries.
    return [
        chunk if i == 0 else f"{chunks[i - 1][-overlap:]} {chunk}"
        for i, chunk in enumerate(chunks)
    ]


def load_chunks(folder: str) -> list[dict]:
    root = Path(folder)
    pdf_files = sorted(root.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in '{folder}'")

    chunks = []
    for pdf_path in pdf_files:
        rel_path = pdf_path.relative_to(root)
        print(f"-> Parsing {rel_path}")
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"   Skipped {rel_path}: {e}")
            continue

        for page in pages:
            for idx, text in enumerate(chunk_text(page["text"])):
                chunks.append({
                    "text": text,
                    "document": str(rel_path),
                    "filename": pdf_path.name,
                    "folder": str(rel_path.parent),
                    "page": page["page"],
                    "chunk_index": idx,
                })

    print(f"Generated {len(chunks)} chunks")
    return chunks
