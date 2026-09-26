import json
import re
from pathlib import Path


CHUNK_PATTERN = re.compile(
    r"^(?P<document>.+)_p(?P<page>\d+)_c\d+$"
)

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "category",
    "question_type",
    "answerable",
    "expected_document",
    "expected_pages",
    "relevant_chunk_ids",
    "reference_answer",
}


def load_evaluation_dataset(path: Path) -> list[dict]:
    """Load and validate a development or test dataset."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    questions = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "Evaluation dataset must be a non-empty list."
        )

    seen_ids = set()

    for position, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {position} must be an object."
            )

        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(
                f"Question {position} is missing: {sorted(missing)}"
            )

        question_id = item["question_id"]

        for field in ("question_id", "question", "category"):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{question_id}: invalid {field}"
                )

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {question_id}"
            )
        seen_ids.add(question_id)

        question_type = item["question_type"]
        if question_type is not None and (
            not isinstance(question_type, str)
            or not question_type.strip()
        ):
            raise ValueError(
                f"{question_id}: invalid question_type"
            )

        if type(item["answerable"]) is not bool:
            raise ValueError(
                f"{question_id}: answerable must be boolean"
            )

        pages = item["expected_pages"]
        chunk_ids = item["relevant_chunk_ids"]

        if not isinstance(pages, list) or any(
            type(page) is not int or page < 1
            for page in pages
        ):
            raise ValueError(
                f"{question_id}: invalid expected_pages"
            )

        if len(pages) != len(set(pages)):
            raise ValueError(
                f"{question_id}: duplicate expected pages"
            )

        if not isinstance(chunk_ids, list) or any(
            not isinstance(chunk, str)
            for chunk in chunk_ids
        ):
            raise ValueError(
                f"{question_id}: invalid relevant_chunk_ids"
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                f"{question_id}: duplicate relevant chunks"
            )

        reference = item["reference_answer"]
        if reference is not None and (
            not isinstance(reference, str)
            or not reference.strip()
        ):
            raise ValueError(
                f"{question_id}: invalid reference_answer"
            )

        if item["answerable"]:
            document = item["expected_document"]

            if (
                not isinstance(document, str)
                or not document.strip()
                or not pages
                or not chunk_ids
            ):
                raise ValueError(
                    f"{question_id}: answerable questions "
                    "require a document, pages and chunk labels"
                )

            for chunk_id in chunk_ids:
                match = CHUNK_PATTERN.fullmatch(chunk_id)

                if (
                    match is None
                    or match.group("document") != document
                    or int(match.group("page")) not in pages
                ):
                    raise ValueError(
                        f"{question_id}: inconsistent "
                        f"chunk label {chunk_id}"
                    )

        else:
            if (
                item["expected_document"] is not None
                or pages
                or chunk_ids
                or reference is not None
            ):
                raise ValueError(
                    f"{question_id}: unanswerable questions "
                    "must not have positive evidence labels"
                )

    return questions