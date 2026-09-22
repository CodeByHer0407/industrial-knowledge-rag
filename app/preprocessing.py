import re


# Matches the specific standalone page-header pattern
# observed in the motor sourcebook.
HEADER_ONLY_PATTERN = re.compile(
    r"(?:APPENDICES\s+)?"
    r"\d+\s*\|\s*"
    r"IMPROVING MOTOR AND DRIVE SYSTEM PERFORMANCE:"
    r"\s*A SOURCEBOOK FOR INDUSTRY",
    flags=re.IGNORECASE,
)


def is_header_only_chunk(text: str) -> bool:
    """
    Identify standalone boilerplate chunks.

    Only matches chunks consisting entirely of the
    known page-header pattern.
    """

    normalized_text = " ".join(text.split())

    return bool(
        HEADER_ONLY_PATTERN.fullmatch(normalized_text)
    )


def filter_header_only_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove standalone boilerplate chunks while
    preserving all remaining chunk IDs and metadata.
    """

    return [
        chunk
        for chunk in chunks
        if not is_header_only_chunk(chunk["text"])
    ]