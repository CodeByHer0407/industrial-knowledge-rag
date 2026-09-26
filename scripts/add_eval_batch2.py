import json
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "dev_questions.json"
DOCUMENT = "motor_manual.pdf"


def make_question(
    question_id, question, category, question_type,
    page, chunks, reference_answer
):
    return {
        "question_id": question_id,
        "question": question,
        "category": category,
        "question_type": question_type,
        "answerable": True,
        "expected_document": DOCUMENT,
        "expected_pages": [page],
        "relevant_chunk_ids": [
            f"{DOCUMENT}_{chunk}" for chunk in chunks
        ],
        "reference_answer": reference_answer,
    }


NEW_QUESTIONS = [
    make_question(
        "Q018",
        "Why can using discharge dampers instead of motor speed "
        "control increase the operating cost of variable-flow fan systems?",
        "fan_systems",
        "factual",
        14,
        ["p14_c0"],
        "Discharge dampers restrict airflow rather than adjusting "
        "motor speed. They are among the least efficient flow-control "
        "methods, so fan systems can consume more energy and have "
        "higher operating costs than necessary.",
    ),
    make_question(
        "Q019",
        "How do eddy-current and hydraulic couplings control "
        "output speed, and what drawbacks do they share?",
        "speed_control",
        "comparison",
        21,
        ["p21_c1"],
        "Eddy-current couplings adjust magnetic field strength "
        "to control slip and rotational speed. Hydraulic couplings "
        "allow fluid to recirculate instead of performing mechanical "
        "work. Both have relatively low efficiency compared with "
        "other speed-control devices and high maintenance costs.",
    ),
    make_question(
        "Q020",
        "What problems can result from installing a fan that "
        "is larger than the system requires?",
        "fan_systems",
        "factual",
        26,
        ["p26_c0"],
        "Oversized fans can increase operating costs, produce "
        "unnecessary noise, and require more maintenance.",
    ),
    make_question(
        "Q021",
        "How can VFDs improve fan operation when airflow "
        "requirements change, and what starting benefit "
        "can they provide?",
        "fan_systems",
        "multi_part",
        26,
        ["p26_c1"],
        "VFDs adjust fan speed and output to meet changing "
        "airflow requirements. They are often an efficient "
        "alternative to dampers and inlet guide vanes. "
        "Their soft-start capabilities can also limit "
        "motor starting currents.",
    ),
    make_question(
        "Q022",
        "Which four electrical quantities can an electrician "
        "measure at a motor control center to help assess "
        "motor-system performance?",
        "motor_assessment",
        "factual",
        33,
        ["p33_c0"],
        "Input kilowatts, voltage, current, and power factor.",
    ),
    make_question(
        "Q023",
        "Why is a load-duty cycle useful when evaluating "
        "motor-system improvements, and what data is it "
        "developed from?",
        "motor_assessment",
        "factual",
        33,
        ["p33_c1"],
        "Motor loads can vary with weather, production demand, "
        "seasons, and product mix. A load-duty cycle developed "
        "from power-logging data indicates how much time "
        "a motor operates at different loads, helping "
        "evaluate improvement opportunities.",
    ),
    make_question(
        "Q024",
        "What annual motor maintenance activities are "
        "listed in the sourcebook's common inspection table?",
        "motor_maintenance",
        "factual",
        39,
        ["p39_c1"],
        "Annual activities include regreasing antifriction "
        "bearings, checking the air gap and bearing clearances, "
        "and cleaning undercut slots in the commutator.",
    ),
    make_question(
        "Q025",
        "What electrical equipment can generate harmonics, "
        "how can facilities reduce their effects, and how "
        "can harmonics affect motor windings?",
        "power_quality",
        "multi_part",
        53,
        ["p53_c1"],
        "Large nonlinear loads, including welders and VFDs, "
        "can generate harmonics. Facilities may use filtering "
        "devices and isolation transformers to reduce their "
        "effects. Harmonics can increase heat generated in "
        "motor windings at a given load.",
    ),
    {
        "question_id": "Q026",
        "question": "What is the exact electricity tariff "
                    "at the reader's factory in September 2026?",
        "category": "out_of_scope",
        "question_type": "unanswerable",
        "answerable": False,
        "expected_document": None,
        "expected_pages": [],
        "relevant_chunk_ids": [],
        "reference_answer": None,
    },
]


def main():
    questions = load_evaluation_dataset(DATASET_PATH)

    existing_ids = {q["question_id"] for q in questions}
    new_ids = {q["question_id"] for q in NEW_QUESTIONS}

    if len(new_ids) != len(NEW_QUESTIONS):
        raise SystemExit("Duplicate IDs within Batch 2.")

    duplicates = existing_ids & new_ids

    if duplicates:
        raise SystemExit(
            f"Already present: {sorted(duplicates)}. "
            "No changes made."
        )

    updated = questions + NEW_QUESTIONS

    temp_path = DATASET_PATH.with_name(
        "dev_questions.batch2.tmp.json"
    )
    temp_path.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    load_evaluation_dataset(temp_path)
    temp_path.replace(DATASET_PATH)

    print(f"Added questions: {len(NEW_QUESTIONS)}")
    print(f"Total development questions: {len(updated)}")
    print(
        "Answerable:",
        sum(q["answerable"] for q in updated),
    )
    print(
        "Unanswerable:",
        sum(not q["answerable"] for q in updated),
    )


if __name__ == "__main__":
    main()