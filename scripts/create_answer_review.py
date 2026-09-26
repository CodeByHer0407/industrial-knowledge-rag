import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT / "eval" / "runs" / "dev_ollama_pilot.json"
)
DEFAULT_OUTPUT = (
    ROOT / "eval" / "runs" / "dev_ollama_pilot_review.csv"
)


def main():
    parser = argparse.ArgumentParser(
        description="Create a manual review worksheet for RAG answers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    report = json.loads(
        args.input.read_text(encoding="utf-8")
    )

    if report.get("dataset") != "dev":
        parser.error("Only development-set runs are supported.")

    results = report["results"]

    fields = [
        "question_id",
        "question",
        "answerable",
        "reference_answer",
        "generated_answer",
        "answer_status",
        "citation_reference_valid",
        "retrieved_evidence",
        "answer_correctness",
        "evidence_support",
        "citation_support",
        "review_notes",
    ]

    rows = []
    answered = 0
    valid_references = 0
    unanswerable = 0
    abstained = 0

    for item in results:
        status = item["answer_status"]
        validation = item["citation_validation"]

        if status == "answered":
            answered += 1
            if validation and validation.get("is_valid"):
                valid_references += 1

        if not item["answerable"]:
            unanswerable += 1
            if status == "abstained":
                abstained += 1

        evidence = []

        for source in item["retrieved_sources"]:
            evidence.append(
                f"RANK {source['rank']} | "
                f"{source['chunk_id']} | "
                f"Page {source['page']}\n"
                f"{source.get('text') or '[TEXT MISSING]'}"
            )

        rows.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "reference_answer": item.get("reference_answer") or "",
            "generated_answer": item["generated_answer"],
            "answer_status": status,
            "citation_reference_valid": (
                validation.get("is_valid")
                if validation is not None
                else "not_applicable"
            ),
            "retrieved_evidence": "\n\n".join(evidence),
            "answer_correctness": "",
            "evidence_support": "",
            "citation_support": "",
            "review_notes": "",
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Review rows created: {len(rows)}")
    print(
        "Valid citation references among answered questions:",
        f"{valid_references}/{answered}",
    )
    print(
        "Abstentions among unanswerable questions:",
        f"{abstained}/{unanswerable}",
    )
    print(f"Saved worksheet: {args.output}")
    print(
        "Answer correctness, grounding and citation support "
        "still require manual review."
    )


if __name__ == "__main__":
    main()