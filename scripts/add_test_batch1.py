import json
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "eval" / "test_questions.json"
DOCUMENT = "motor_manual.pdf"


# ID, question, category, type, pages, chunks, reference answer
QUESTIONS = [
    (
        "Q043",
        "What speed adjustment range can DC motors provide, "
        "and how low can their operating speed be relative "
        "to their base speed?",
        "speed_control",
        "numerical",
        [42],
        ["p42_c1"],
        "DC motors can provide speed adjustment ratios as "
        "high as 20:1. They can operate at approximately "
        "5% to 7% of base speed, and some can operate "
        "at zero RPM.",
    ),
    (
        "Q044",
        "How does a NEMA Design A motor differ from a "
        "comparable Design B motor, and what kind of "
        "application might require Design A?",
        "motor_selection",
        "comparison",
        [44],
        ["p44_c1"],
        "Design A motors generally have higher breakdown "
        "torque, higher starting current, and less slip "
        "than Design B motors. They may be selected for "
        "applications requiring large transient increases "
        "in load, such as crushers or injection molding machines.",
    ),
    (
        "Q045",
        "Which factors determine the voltage and enclosure "
        "specified for an industrial motor, and how does "
        "the sourcebook classify low- and medium-voltage motors?",
        "motor_selection",
        "multi_part",
        [45],
        ["p45_c0"],
        "Motor voltage is determined by the plant's available "
        "power supply, while enclosure selection depends "
        "on environmental conditions such as air quality, "
        "moisture exposure, and harmful vapors. The sourcebook "
        "calls motors rated at 600 V or below low-voltage "
        "and describes 2,300 V and 4,000 V motors as medium-voltage.",
    ),
    (
        "Q046",
        "What are the two principal approaches to adjusting "
        "the speed of motor-driven equipment, and what "
        "drawbacks can intermediate mechanical devices introduce?",
        "speed_control",
        "comparison",
        [48],
        ["p48_c0", "p48_c1"],
        "Speed can be adjusted directly at the motor or "
        "by using a constant-speed motor with an intermediate "
        "device that changes the speed ratio. Intermediate "
        "devices such as gearing or adjustable-pitch pulley "
        "systems introduce additional components, which can "
        "increase failure risk and efficiency losses.",
    ),
    (
        "Q047",
        "When a service center repairs motor windings, "
        "which original coil characteristics should it "
        "reproduce, and what wire-supply capability is useful?",
        "motor_repair",
        "multi_part",
        [56],
        ["p56_c1"],
        "Unless better alternatives have been agreed upon, "
        "the repair should reproduce the original coil "
        "dimensions, number of effective turns, and "
        "cross-sectional area. The service center should "
        "stock a broad range of wire sizes or be able "
        "to obtain the required sizes quickly.",
    ),
    (
        "Q048",
        "According to the sourcebook's example, for a "
        "corporation with a 10% profit margin, how much "
        "sales revenue is equivalent to saving one dollar "
        "in electricity costs?",
        "motor_economics",
        "numerical",
        [60],
        ["p60_c0"],
        "For a company with a 10% profit margin, one dollar "
        "saved in electricity costs is equivalent to "
        "ten dollars in sales revenue.",
    ),
    (
        "Q049",
        "What information does the sourcebook recommend "
        "including in a proposal for an industrial "
        "energy-efficiency improvement project?",
        "motor_economics",
        "multi_part",
        [60],
        ["p60_c0", "p60_c1"],
        "The proposal should identify specific efficiency "
        "opportunities, provide life-cycle cost results "
        "for proposed projects, identify projects with "
        "the greatest net benefits, and connect project "
        "benefits with corporate financial priorities "
        "and current needs, including relevant energy "
        "or greenhouse-gas reduction goals.",
    ),
]


def main():
    # Do not overwrite or modify an existing held-out dataset.
    if TEST_PATH.exists():
        raise SystemExit(
            "test_questions.json already exists. "
            "Review it before making any changes."
        )

    records = []

    for (
        question_id,
        question,
        category,
        question_type,
        pages,
        chunks,
        reference_answer,
    ) in QUESTIONS:
        records.append({
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

    temp_path = TEST_PATH.with_name(
        "test_questions.batch1.tmp.json"
    )

    temp_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Validate schema before creating the held-out file.
    load_evaluation_dataset(temp_path)
    temp_path.replace(TEST_PATH)

    print(f"Created: {TEST_PATH}")
    print(f"Total test questions: {len(records)}")
    print("Answerable: 7")
    print("Unanswerable: 0")


if __name__ == "__main__":
    main()