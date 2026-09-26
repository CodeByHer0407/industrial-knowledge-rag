import json
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "dev_questions.json"
DOCUMENT = "motor_manual.pdf"


ANSWERABLE = [
    (
        "Q037",
        "How does a wound-rotor induction motor control speed "
        "and torque, and what operating advantages does it provide?",
        "motor_types",
        "multi_part",
        [18],
        ["p18_c0", "p18_c1"],
        "A wound-rotor induction motor controls speed and torque "
        "by varying external resistance in its rotor circuit. "
        "Its advantages include excellent speed control, "
        "high starting torque, low starting current, the ability "
        "to handle high-inertia loads and frequent starts and "
        "stops, and prolonged operation at reduced speeds.",
    ),
    (
        "Q038",
        "How do motor circuit analysis, motor-current signature "
        "analysis, and electrical signature analysis differ "
        "in the conditions and signals they examine?",
        "motor_diagnostics",
        "comparison",
        [41],
        ["p41_c0", "p41_c1"],
        "Motor circuit analysis evaluates winding and ground "
        "insulation and the rotor while equipment is de-energized. "
        "Motor-current signature analysis examines current "
        "spectra while energized to identify rotor, air-gap "
        "and load-related faults. Electrical signature analysis "
        "examines both voltage and current spectra while "
        "energized to identify current-related and supply faults.",
    ),
    (
        "Q039",
        "What plant distribution-system conditions can cause "
        "power-quality problems, and what preventive or "
        "investigative actions does the sourcebook recommend?",
        "power_quality",
        "multi_part",
        [54],
        ["p54_c1"],
        "Adding single-phase loads to one phase can create "
        "voltage unbalance, while deterioration of grounding "
        "can disrupt equipment or fault-clearing devices. "
        "The sourcebook recommends checking grounding adequacy, "
        "reviewing system capacity before adding loads, and "
        "performing a power-quality review when symptoms "
        "such as overheating or frequent voltage sags occur.",
    ),
]


UNANSWERABLE = [
    (
        "Q040",
        "What is the real-time power consumption of the "
        "main motor currently installed at my factory?",
    ),
    (
        "Q041",
        "What warranty period applies to the specific "
        "wound-rotor motor installed at my plant?",
    ),
    (
        "Q042",
        "What was the measured total harmonic distortion "
        "at my facility during its latest electrical inspection?",
    ),
]


def main():
    questions = load_evaluation_dataset(DATASET_PATH)
    new_questions = []

    for (
        question_id,
        question,
        category,
        question_type,
        pages,
        chunks,
        reference_answer,
    ) in ANSWERABLE:
        new_questions.append({
            "question_id": question_id,
            "question": question,
            "category": category,
            "question_type": question_type,
            "answerable": True,
            "expected_document": DOCUMENT,
            "expected_pages": pages,
            "relevant_chunk_ids": [
                f"{DOCUMENT}_{chunk}" for chunk in chunks
            ],
            "reference_answer": reference_answer,
        })

    for question_id, question in UNANSWERABLE:
        new_questions.append({
            "question_id": question_id,
            "question": question,
            "category": "out_of_scope",
            "question_type": "unanswerable",
            "answerable": False,
            "expected_document": None,
            "expected_pages": [],
            "relevant_chunk_ids": [],
            "reference_answer": None,
        })

    existing_ids = {q["question_id"] for q in questions}
    new_ids = [q["question_id"] for q in new_questions]

    if len(set(new_ids)) != len(new_ids):
        raise SystemExit("Duplicate IDs within Batch 4.")

    duplicates = existing_ids.intersection(new_ids)

    if duplicates:
        raise SystemExit(
            f"Questions already present: {sorted(duplicates)}. "
            "No changes made."
        )

    updated = questions + new_questions

    temp_path = DATASET_PATH.with_name(
        "dev_questions.batch4.tmp.json"
    )

    temp_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Validate everything before replacing the dataset.
    load_evaluation_dataset(temp_path)
    temp_path.replace(DATASET_PATH)

    print(f"Added questions: {len(new_questions)}")
    print(f"Total development questions: {len(updated)}")
    print("Answerable:", sum(q["answerable"] for q in updated))
    print(
        "Unanswerable:",
        sum(not q["answerable"] for q in updated),
    )


if __name__ == "__main__":
    main()