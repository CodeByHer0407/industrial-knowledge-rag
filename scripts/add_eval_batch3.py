import json
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "dev_questions.json"
DOCUMENT = "motor_manual.pdf"


# ID, question, category, type, pages, chunks, reference answer
ANSWERABLE = [
    (
        "Q027",
        "What limitations make single-phase motors less suitable "
        "for many industrial applications, and what advantages "
        "do polyphase motors offer?",
        "motor_types",
        "comparison",
        [15],
        ["p15_c0", "p15_c1"],
        "Integral single-phase induction motors can draw high "
        "starting currents relative to their size, tend to be "
        "less efficient than comparable three-phase motors, "
        "and have more limited speed and power-rating options. "
        "Polyphase motors offer high efficiency, favorable "
        "torque and current characteristics, and relatively "
        "low cost, making them widely used in industry.",
    ),
    (
        "Q028",
        "What is motor slip, how does it distinguish induction "
        "motors from synchronous motors, and what happens "
        "to slip when the motor load increases?",
        "motor_characteristics",
        "multi_part",
        [17],
        ["p17_c0", "p17_c1"],
        "Slip is the difference between actual motor speed "
        "and synchronous speed. Induction motors operate "
        "slightly below synchronous speed, while synchronous "
        "motors operate without slip. In an induction motor, "
        "slip increases as the load increases.",
    ),
    (
        "Q029",
        "Which factors determine the operating speed of "
        "DC motors and AC motors?",
        "motor_characteristics",
        "comparison",
        [20],
        ["p20_c0"],
        "DC motor speed depends on the motor type, magnetic "
        "field strength, and load. AC motor speed depends "
        "on rotor type, number of poles, supply frequency, "
        "and slip characteristics.",
    ),
    (
        "Q030",
        "In the sourcebook's illustrative pumping example, "
        "how does energy use change when an adjustable-speed "
        "drive replaces control-valve-based flow control?",
        "pumping_systems",
        "numerical",
        [24],
        ["p24_c0"],
        "The illustration compares 100 energy units supplied "
        "to the original system with 80 units for the "
        "adjustable-speed-drive system doing the same work. "
        "The drive reduces pump flow and energy lost across "
        "the throttle valve.",
    ),
    (
        "Q031",
        "What are five potential consequences of choosing "
        "a motor that is larger than the application requires?",
        "motor_sizing",
        "factual",
        [28],
        ["p28_c1"],
        "The sourcebook lists lower efficiency, higher "
        "motor and controller costs, higher installation "
        "costs, lower power factor, and increased "
        "operating costs.",
    ),
    (
        "Q032",
        "Which costs and operating factors should be "
        "considered when comparing motor repair and "
        "replacement using life-cycle cost analysis?",
        "financial_analysis",
        "factual",
        [35],
        ["p35_c0"],
        "The comparison should consider operating hours "
        "and electricity costs, as well as motor purchase "
        "and repair costs, rather than relying only "
        "on initial cost.",
    ),
    (
        "Q033",
        "How can moisture, contaminated lubricant, and "
        "other contaminants damage motor winding insulation?",
        "motor_maintenance",
        "multi_part",
        [38],
        ["p38_c0", "p38_c1"],
        "Moisture reduces insulation's dielectric strength "
        "and increases the risk of sudden failure. "
        "Contaminated lubricant can have reduced dielectric "
        "strength and encourage contaminant accumulation. "
        "Contaminants on shifting windings can also cause "
        "abrasive wear and early insulation failure.",
    ),
    (
        "Q034",
        "What can cause motor bearings to fail, and which "
        "operating conditions should be considered when "
        "selecting suitable bearings?",
        "motor_maintenance",
        "multi_part",
        [46],
        ["p46_c0"],
        "Bearings can fail when lubrication breaks down, "
        "lubrication is lost, or solid contaminants enter. "
        "Bearing selection should account for load, "
        "temperature, environmental conditions, speed, "
        "coupling method, lubrication method, and the "
        "frequency of motor starts and stops.",
    ),
]


UNANSWERABLE = [
    (
        "Q035",
        "What is the serial number of the main electric "
        "motor installed at my factory?",
    ),
    (
        "Q036",
        "What is the current measured bearing temperature "
        "of the main motor at my factory?",
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

    existing_ids = {
        item["question_id"] for item in questions
    }
    new_ids = [
        item["question_id"] for item in new_questions
    ]

    if len(set(new_ids)) != len(new_ids):
        raise SystemExit("Duplicate IDs within Batch 3.")

    duplicates = existing_ids.intersection(new_ids)

    if duplicates:
        raise SystemExit(
            f"Already present: {sorted(duplicates)}. "
            "No changes made."
        )

    updated = questions + new_questions

    # Validate the complete updated dataset before saving.
    temp_path = DATASET_PATH.with_name(
        "dev_questions.batch3.tmp.json"
    )

    temp_path.write_text(
        json.dumps(
            updated,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    load_evaluation_dataset(temp_path)
    temp_path.replace(DATASET_PATH)

    print(f"Added questions: {len(new_questions)}")
    print(f"Total development questions: {len(updated)}")
    print(
        "Answerable:",
        sum(item["answerable"] for item in updated),
    )
    print(
        "Unanswerable:",
        sum(not item["answerable"] for item in updated),
    )


if __name__ == "__main__":
    main()