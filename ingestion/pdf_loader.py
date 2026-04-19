from pypdf import PdfReader
from pathlib import Path
from loguru import logger


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load PDF and return list of pages with metadata.
    Each page is a dict with 'text', 'page_number', 'source'.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    reader = PdfReader(str(path))
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_number": page_num + 1,
                "source": path.name
            })

    logger.info(f"Loaded {len(pages)} pages from {path.name}")
    return pages


if __name__ == "__main__":
    pages = load_pdf("data/budget_2024.pdf")
    print(f"\nTotal pages loaded: {len(pages)}")
    print(f"\nFirst page preview (first 300 chars):")
    print(pages[0]["text"][:300])
    print(f"\nLast page preview (first 300 chars):")
    print(pages[-1]["text"][:300])