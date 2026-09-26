import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "eval" / "runs" /
    "dev_ollama_full_review_labeled.csv"
)

OUTPUT = (
    ROOT / "eval" / "reports" /
    "dev_ollama_answer_quality.json"
)

REQUIRED_FIELDS = [
    "question_id",
    "answerable",
    "answer_status",
    "citation_reference_valid",
    "answer_correctness",
    "evidence_support",
    "citation_support",
    "review_notes",
]


def percentage(numerator, denominator):
    if denominator == 0:
        return None
    return round(100 * numerator / denominator, 2)


def main():
    with INPUT.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not set(REQUIRED_FIELDS).issubset(
            reader.fieldnames or []
        ):
            raise SystemExit("CSV is missing required columns.")

        rows = list(reader)

    if len(rows) != 42:
        raise SystemExit(
            f"Expected 42 questions; found {len(rows)}."
        )

    ids = [row["question_id"] for row in rows]

    if len(set(ids)) != 42:
        raise SystemExit("Duplicate question IDs found.")

    allowed = {
        "answer_correctness": {
            "complete",
            "partial",
            "incorrect",
            "not_applicable",
        },
        "evidence_support": {
            "supported",
            "partially_supported",
            "unsupported",
            "not_applicable",
        },
        "citation_support": {
            "supported",
            "partially_supported",
            "unsupported",
            "not_applicable",
        },
    }

    for row in rows:
        for field, values in allowed.items():
            if row[field].strip() not in values:
                raise SystemExit(
                    f"{row['question_id']}: "
                    f"invalid or missing {field}"
                )

        if not row["review_notes"].strip():
            raise SystemExit(
                f"{row['question_id']}: missing review notes"
            )

    answerable = [
        row for row in rows
        if row["answerable"].lower() == "true"
    ]

    unanswerable = [
        row for row in rows
        if row["answerable"].lower() == "false"
    ]

    if len(answerable) != 35 or len(unanswerable) != 7:
        raise SystemExit(
            "Unexpected development dataset composition."
        )

    answered = [
        row for row in rows
        if row["answer_status"] == "answered"
    ]

    correct_abstentions = [
        row for row in unanswerable
        if row["answer_status"] == "abstained"
    ]

    incorrect_abstentions = [
        row for row in answerable
        if row["answer_status"] == "abstained"
    ]

    correctness = Counter(
        row["answer_correctness"]
        for row in answerable
    )

    evidence_support = Counter(
        row["evidence_support"]
        for row in answered
    )

    citation_support = Counter(
        row["citation_support"]
        for row in answered
    )

    valid_citation_references = [
        row for row in answered
        if row["citation_reference_valid"].lower() == "true"
    ]

    report = {
        "dataset": "dev",
        "evaluation_type": "manual_answer_review",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "total_questions": len(rows),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "answer_correctness": {
            "complete": correctness["complete"],
            "partial": correctness["partial"],
            "incorrect": correctness["incorrect"],
            "complete_rate_pct": percentage(
                correctness["complete"],
                len(answerable),
            ),
        },
        "abstention": {
            "correct": len(correct_abstentions),
            "incorrect": len(incorrect_abstentions),
            "correct_rate_pct": percentage(
                len(correct_abstentions),
                len(unanswerable),
            ),
            "incorrect_question_ids": [
                row["question_id"]
                for row in incorrect_abstentions
            ],
        },
        "citation_reference_validation": {
            "valid": len(valid_citation_references),
            "answered_questions": len(answered),
            "valid_rate_pct": percentage(
                len(valid_citation_references),
                len(answered),
            ),
            "failed_question_ids": [
                row["question_id"]
                for row in answered
                if row["citation_reference_valid"].lower()
                != "true"
            ],
        },
        "manual_evidence_support": dict(evidence_support),
        "manual_citation_support": dict(citation_support),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nANSWER-LEVEL EVALUATION")
    print("=" * 45)

    print("Total questions:", len(rows))
    print("Answerable:", len(answerable))
    print("Unanswerable:", len(unanswerable))

    print("\nAnswer correctness:")
    print("Complete:", correctness["complete"])
    print("Partial:", correctness["partial"])
    print("Incorrect:", correctness["incorrect"])

    print("\nAbstention:")
    print(
        "Correct:",
        f"{len(correct_abstentions)}/{len(unanswerable)}",
    )
    print(
        "Incorrect question IDs:",
        report["abstention"]["incorrect_question_ids"],
    )

    print("\nCitation-reference validation:")
    print(
        "Valid:",
        f"{len(valid_citation_references)}/{len(answered)}",
    )
    print(
        "Failed question IDs:",
        report["citation_reference_validation"][
            "failed_question_ids"
        ],
    )

    print("\nManual evidence support:")
    print(dict(evidence_support))

    print("\nManual citation support:")
    print(dict(citation_support))

    print("\nReport saved:", OUTPUT)


if __name__ == "__main__":
    main()