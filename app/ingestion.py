from pathlib import Path

import pymupdf


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text and metadata from every non-empty PDF page.
    """

    pdf_path = Path(pdf_path)

    # Validate the input file
    if not pdf_path.is_file():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file.")

    extracted_pages = []

    # Open the PDF
    with pymupdf.open(pdf_path) as document:

        # Process each page
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            # Skip pages without extractable text
            if not text:
                continue

            # Store extracted text and metadata
            extracted_pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text
            })

    return extracted_pages