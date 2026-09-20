def chunk_documents(
    pages: list[dict],
    chunk_size: int = 120,
    chunk_overlap: int = 25,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping word-based chunks.

    Each chunk preserves its original source and page number.
    """

    # Step 1: Validate configuration
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and less than chunk_size."
        )

    chunks = []

    # Step 2: Process each extracted PDF page
    for page_data in pages:

        text = page_data["text"]
        source = page_data["source"]
        page_number = page_data["page"]

        # Split text into individual words
        words = text.split()

        if not words:
            continue

        # Step 3: Calculate the movement between chunks
        step = chunk_size - chunk_overlap

        chunk_index = 0

        # Step 4: Create overlapping chunks
        for start in range(0, len(words), step):

            end = min(start + chunk_size, len(words))

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            # Step 5: Preserve source metadata
            chunks.append({
                "chunk_id": f"{source}_p{page_number}_c{chunk_index}",
                "source": source,
                "page": page_number,
                "text": chunk_text,
            })

            chunk_index += 1

            # Stop after including the final word
            if end == len(words):
                break

    return chunks