import csv
import json

import pytest

from scripts import summarize_answer_evaluation as evaluator


UNANSWERABLE = {
    "Q017", "Q026", "Q035", "Q036",
    "Q040", "Q041", "Q042",
}

INCORRECT_ABSTENTIONS = {"Q016", "Q024"}

INVALID_CITATIONS = {
    "Q002", "Q014", "Q015",
    "Q021", "Q025", "Q028",
}


def make_review_rows():
    """Create synthetic data matching our evaluation's structure."""
    answered_ids = [
        f"Q{i:03d}"
        for i in range(1, 43)
        if f"Q{i:03d}" not in (
            UNANSWERABLE | INCORRECT_ABSTENTIONS
        )
    ]

    rows = []

    for i in range(1, 43):
        question_id = f"Q{i:03d}"
        answerable = question_id not in UNANSWERABLE
        abstained = (
            not answerable
            or question_id in INCORRECT_ABSTENTIONS
        )

        if not answerable:
            correctness = "not_applicable"
        elif abstained:
            correctness = "incorrect"
        elif question_id in answered_ids[:16]:
            correctness = "complete"
        else:
            correctness = "partial"

        if abstained:
            evidence = "not_applicable"
            citation = "not_applicable"
            reference_valid = "not_applicable"
        else:
            position = answered_ids.index(question_id)

            evidence = (
                "partially_supported"
                if question_id == "Q013"
                else "supported"
            )

            if position < 23:
                citation = "supported"
            elif position < 31:
                citation = "partially_supported"
            else:
                citation = "unsupported"

            reference_valid = str(
                question_id not in INVALID_CITATIONS
            )

        rows.append({
            "question_id": question_id,
            "answerable": str(answerable),
            "answer_status": (
                "abstained" if abstained else "answered"
            ),
            "citation_reference_valid": reference_valid,
            "answer_correctness": correctness,
            "evidence_support": evidence,
            "citation_support": citation,
            "review_notes": "Synthetic test fixture.",
        })

    return rows


def configure_paths(tmp_path, monkeypatch, rows):
    input_path = tmp_path / "review.csv"
    output_path = tmp_path / "report.json"

    with input_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=evaluator.REQUIRED_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(evaluator, "INPUT", input_path)
    monkeypatch.setattr(evaluator, "OUTPUT", output_path)

    return output_path


def test_generates_correct_report(tmp_path, monkeypatch):
    output_path = configure_paths(
        tmp_path, monkeypatch, make_review_rows()
    )

    evaluator.main()

    report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert report["total_questions"] == 42
    assert report["answerable_questions"] == 35
    assert report["unanswerable_questions"] == 7

    assert report["answer_correctness"]["complete"] == 16
    assert report["answer_correctness"]["partial"] == 17
    assert report["answer_correctness"]["incorrect"] == 2

    assert report["abstention"]["correct"] == 7
    assert set(
        report["abstention"]["incorrect_question_ids"]
    ) == INCORRECT_ABSTENTIONS

    citations = report["citation_reference_validation"]
    assert citations["valid"] == 27
    assert citations["answered_questions"] == 33
    assert set(
        citations["failed_question_ids"]
    ) == INVALID_CITATIONS

    assert report["manual_evidence_support"] == {
        "supported": 32,
        "partially_supported": 1,
    }

    assert report["manual_citation_support"] == {
        "supported": 23,
        "partially_supported": 8,
        "unsupported": 2,
    }


def test_rejects_missing_review_label(
    tmp_path, monkeypatch
):
    rows = make_review_rows()
    rows[0]["answer_correctness"] = ""

    configure_paths(tmp_path, monkeypatch, rows)

    with pytest.raises(
        SystemExit, match="invalid or missing"
    ):
        evaluator.main()


def test_rejects_duplicate_question_ids(
    tmp_path, monkeypatch
):
    rows = make_review_rows()
    rows[-1]["question_id"] = rows[0]["question_id"]

    configure_paths(tmp_path, monkeypatch, rows)

    with pytest.raises(
        SystemExit, match="Duplicate question IDs"
    ):
        evaluator.main()


def test_rejects_missing_review_notes(
    tmp_path, monkeypatch
):
    rows = make_review_rows()
    rows[0]["review_notes"] = ""

    configure_paths(tmp_path, monkeypatch, rows)

    with pytest.raises(
        SystemExit, match="missing review notes"
    ):
        evaluator.main()