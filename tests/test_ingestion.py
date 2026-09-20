import pytest
import pymupdf

from app.ingestion import extract_pdf_pages


def test_extract_pdf_pages(tmp_path):
    """Verify text extraction and metadata."""

    pdf_path = tmp_path / "sample.pdf"

    # Create a temporary PDF
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Motor maintenance instructions."
        )
        document.save(pdf_path)

    # Execute our extraction function
    result = extract_pdf_pages(pdf_path)

    # Verify the output
    assert len(result) == 1
    assert result[0]["source"] == "sample.pdf"
    assert result[0]["page"] == 1
    assert "Motor maintenance" in result[0]["text"]


def test_empty_pdf_page(tmp_path):
    """Verify that blank pages are skipped."""

    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(pdf_path)

    result = extract_pdf_pages(pdf_path)

    assert result == []


def test_missing_pdf(tmp_path):
    """Verify missing-file error handling."""

    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(pdf_path)


def test_invalid_file_type(tmp_path):
    """Verify non-PDF files are rejected."""

    file_path = tmp_path / "document.txt"
    file_path.write_text("Sample text")

    with pytest.raises(ValueError):
        extract_pdf_pages(file_path)