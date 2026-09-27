import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    BM25 keyword retriever over the same chunks used by FAISS.

    Uses Okapi BM25 with k1=1.5 and b=0.75.
    """

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self.term_frequencies = []
        self.document_lengths = []

        seen_ids = set()

        for chunk in self.chunks:
            for field in ("chunk_id", "source", "page", "text"):
                if field not in chunk:
                    raise ValueError(
                        f"Chunk is missing required field: {field}"
                    )

            chunk_id = chunk["chunk_id"]

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Chunk ID must be a nonempty string.")

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            if (
                not isinstance(chunk["text"], str)
                or not chunk["text"].strip()
            ):
                raise ValueError(
                    f"Chunk {chunk_id} has no valid text."
                )

            tokens = tokenize(chunk["text"])
            frequencies = Counter(tokens)

            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))

        self.document_count = len(self.chunks)

        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

        document_frequencies = Counter()

        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())

        # Positive BM25 IDF for each term.
        self.idf = {
            term: math.log1p(
                (self.document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return the highest-scoring matching chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not self.chunks:
            return []

        query_terms = set(tokenize(query))
        scored_chunks = []

        for position, frequencies in enumerate(
            self.term_frequencies
        ):
            document_length = self.document_lengths[position]

            normalization = (
                self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )
            )

            score = 0.0

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                score += (
                    self.idf[term]
                    * term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + normalization)
                )

            if score > 0:
                scored_chunks.append((score, position))

        # Highest score first; original chunk order breaks ties.
        scored_chunks.sort(
            key=lambda item: (-item[0], item[1])
        )

        results = []

        for score, position in scored_chunks[:top_k]:
            chunk = self.chunks[position]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            })

        return results