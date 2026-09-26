import sys
from copy import deepcopy

import json
import pytest

from app.eval_dataset import load_evaluation_dataset
from scripts import evaluate_retrieval


BASE_QUESTION = {
    "question_id": "Q001",
    "question": "How does motor speed affect energy consumption?",
    "category": "operating_speed",
    "question_type": None,
    "answerable": True,
    "expected_document": "motor_manual.pdf",
    "expected_pages": [29],
    "relevant_chunk_ids": [
        "motor_manual.pdf_p29_c2"
    ],
    "reference_answer": None,
}


def save_dataset(tmp_path, questions):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(questions),
        encoding="utf-8",
    )
    return path


def test_loads_valid_dataset(tmp_path):
    path = save_dataset(tmp_path, [BASE_QUESTION])

    result = load_evaluation_dataset(path)

    assert len(result) == 1
    assert result[0]["question_id"] == "Q001"


def test_rejects_duplicate_question_ids(tmp_path):
    path = save_dataset(
        tmp_path,
        [BASE_QUESTION, deepcopy(BASE_QUESTION)],
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_evaluation_dataset(path)


def test_rejects_inconsistent_chunk_labels(tmp_path):
    question = deepcopy(BASE_QUESTION)
    question["relevant_chunk_ids"] = [
        "another_manual.pdf_p29_c2"
    ]

    path = save_dataset(tmp_path, [question])

    with pytest.raises(ValueError, match="inconsistent"):
        load_evaluation_dataset(path)


def test_accepts_unanswerable_question(tmp_path):
    question = deepcopy(BASE_QUESTION)
    question.update({
        "answerable": False,
        "expected_document": None,
        "expected_pages": [],
        "relevant_chunk_ids": [],
    })

    path = save_dataset(tmp_path, [question])

    result = load_evaluation_dataset(path)

    assert result[0]["answerable"] is False


def test_rejects_unanswerable_question_with_labels(tmp_path):
    question = deepcopy(BASE_QUESTION)
    question["answerable"] = False

    path = save_dataset(tmp_path, [question])

    with pytest.raises(ValueError, match="unanswerable"):
        load_evaluation_dataset(path)


def test_final_test_requires_confirmation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_retrieval", "--dataset", "test"],
    )

    with pytest.raises(SystemExit) as exc:
        evaluate_retrieval.main()

    assert exc.value.code == 2