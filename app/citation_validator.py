import re


# Expected format: [motor_manual.pdf, p. 29]
CITATION_PATTERN = re.compile(
    r"\[([^\[\]\n]+?),\s*p\.\s*(\d+)\]"
)

# Detect bracketed references, including incorrectly formatted ones.
BRACKET_PATTERN = re.compile(
    r"\[[^\[\]\n]+\]"
)


def validate_citations(
    answer: str,
    retrieved_sources: list[dict],
) -> dict:
    """
    Check whether citations in an LLM answer refer to
    source/page combinations present in retrieved evidence.

    This checks citation references, not factual grounding.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    # Build the set of source/page combinations actually retrieved.
    allowed_sources = {
        (source["source"], int(source["page"]))
        for source in retrieved_sources
    }

    valid_citations = []
    invalid_citations = []

    # Find every bracketed reference in the answer.
    citations = BRACKET_PATTERN.findall(answer)

    for citation in citations:

        match = CITATION_PATTERN.fullmatch(citation)

        # Reject incorrectly formatted citations.
        if match is None:
            invalid_citations.append(citation)
            continue

        source_name = match.group(1)
        page_number = int(match.group(2))

        # Check whether this source/page was retrieved.
        if (source_name, page_number) in allowed_sources:
            valid_citations.append(citation)
        else:
            invalid_citations.append(citation)

    return {
        "has_citations": bool(citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "is_valid": (
            bool(valid_citations)
            and not invalid_citations
        ),
    }