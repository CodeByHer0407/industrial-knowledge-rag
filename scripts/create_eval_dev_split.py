import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "eval" / "questions.json"
OUTPUT_PATH = ROOT / "eval" / "dev_questions.json"

CHUNK_ID_PATTERN = re.compile(
    r"^(?P<source>.+)_p(?P<page>\d+)_c\d+$"
)


def main():
    original_questions = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    dev_questions = []
    seen_ids = set()

    for item in original_questions:
        question_id = item["question_id"]

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {question_id}"
            )

        seen_ids.add(question_id)

        sources = set()
        pages = set()

        for chunk_id in item["relevant_chunk_ids"]:
            match = CHUNK_ID_PATTERN.fullmatch(chunk_id)

            if match is None:
                raise ValueError(
                    f"Unexpected chunk ID format: {chunk_id}"
                )

            sources.add(match.group("source"))
            pages.add(int(match.group("page")))

        if len(sources) != 1:
            raise ValueError(
                f"Expected one source for {question_id}: {sources}"
            )

        dev_questions.append(
            {
                "question_id": question_id,
                "question": item["question"],
                "category": item["category"],
                "question_type": None,
                "answerable": True,
                "expected_document": next(iter(sources)),
                "expected_pages": sorted(pages),
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "reference_answer": None,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            dev_questions,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        f"Migrated {len(dev_questions)} questions "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()