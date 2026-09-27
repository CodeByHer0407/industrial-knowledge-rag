import re


CHUNK_PATTERN = re.compile(r"_p(\d+)_c(\d+)$")


def expand_adjacent_chunks(
    retrieved_chunks: list[dict],
    all_chunks: list[dict],
    max_total_chunks: int = 10,
) -> list[dict]:
    """
    Preserve ranked retrieval results and append adjacent chunks.

    Only chunks on the same page and from the same source
    are considered neighbors. Retrieved ranks are unchanged.
    """

    if max_total_chunks <= 0:
        raise ValueError("max_total_chunks must be positive.")

    if len(retrieved_chunks) > max_total_chunks:
        raise ValueError(
            "max_total_chunks cannot be smaller than "
            "the number of retrieved chunks."
        )

    # Map corpus chunks by source, page and chunk number.
    lookup = {}

    for chunk in all_chunks:
        match = CHUNK_PATTERN.search(chunk["chunk_id"])

        if not match:
            continue

        page = int(match.group(1))
        chunk_number = int(match.group(2))

        if page != int(chunk["page"]):
            raise ValueError(
                f"Chunk page mismatch: {chunk['chunk_id']}"
            )

        key = (
            chunk["source"],
            page,
            chunk_number,
        )

        if key in lookup:
            raise ValueError(
                f"Duplicate chunk position: {key}"
            )

        lookup[key] = chunk

    # Keep every ranked result before adding any neighbors.
    expanded = [dict(chunk) for chunk in retrieved_chunks]

    seen = {
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    }

    for retrieved in retrieved_chunks:
        match = CHUNK_PATTERN.search(
            retrieved["chunk_id"]
        )

        if not match:
            continue

        page = int(match.group(1))
        chunk_number = int(match.group(2))

        if page != int(retrieved["page"]):
            raise ValueError(
                f"Retrieved chunk page mismatch: "
                f"{retrieved['chunk_id']}"
            )

        # Include the immediately preceding and following
        # chunks, provided they belong to the same page.
        for offset in (-1, 1):
            neighbor_key = (
                retrieved["source"],
                page,
                chunk_number + offset,
            )

            neighbor = lookup.get(neighbor_key)

            if neighbor is None:
                continue

            if neighbor["chunk_id"] in seen:
                continue

            if len(expanded) >= max_total_chunks:
                return expanded

            expanded.append({
                "chunk_id": neighbor["chunk_id"],
                "source": neighbor["source"],
                "page": neighbor["page"],
                "text": neighbor["text"],
                # This passage was added through expansion,
                # not independently scored by retrieval.
                "score": None,
                "retrieval_origin": "adjacent_chunk",
            })

            seen.add(neighbor["chunk_id"])

    return expanded