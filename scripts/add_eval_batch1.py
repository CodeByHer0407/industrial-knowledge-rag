import json
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "dev_questions.json"
DOCUMENT = "motor_manual.pdf"


def make_question(
    question_id,
    question,
    category,
    question_type,
    pages,
    chunk_ids,
    reference_answer,
):
    return {
        "question_id": question_id,
        "question": question,
        "category": category,
        "question_type": question_type,
        "answerable": True,
        "expected_document": DOCUMENT,
        "expected_pages": pages,
        "relevant_chunk_ids": [
            f"{DOCUMENT}_{chunk_id}" for chunk_id in chunk_ids
        ],
        "reference_answer": reference_answer,
    }


NEW_QUESTIONS = [
    make_question(
        "Q011",
        "Who should take the industrial electrical measurements "
        "discussed in the sourcebook?",
        "safety",
        "factual",
        [12],
        ["p12_c0"],
        "Only qualified electricians trained in safety practices "
        "for industrial electrical systems should take the "
        "measurements. Unqualified or untrained personnel "
        "should not attempt them.",
    ),
    make_question(
        "Q012",
        "What voltage conditions can cause motor operating "
        "problems, and how much variation from rated voltage "
        "significantly affects performance?",
        "motor_voltage",
        "factual",
        [19],
        ["p19_c1"],
        "A mismatch with supply voltage or distribution-system "
        "problems such as three-phase voltage unbalance, "
        "outages, sags, surges, overvoltage and undervoltage "
        "can cause problems. Motor performance is significantly "
        "affected by voltage variations of plus or minus "
        "10 percent or more from rated voltage.",
    ),
    make_question(
        "Q013",
        "How do VFDs reduce pumping-system energy requirements, "
        "and in which applications may they be unsuitable?",
        "pumping_systems",
        "factual",
        [25],
        ["p25_c1"],
        "VFDs adjust pump speed to match the system's needs. "
        "Reducing speed proportionally reduces flow while "
        "reducing power requirements exponentially. They may "
        "be unsuitable for pumps operating against high "
        "static or elevation head.",
    ),
    make_question(
        "Q014",
        "Which industrial systems tend to be major consumers "
        "of motor energy, and how should individual plants "
        "identify their most energy-intensive motor systems?",
        "energy_assessment",
        "multi_part",
        [32],
        ["p32_c0", "p32_c1"],
        "Industries frequently using pumps, fans, material "
        "handling systems and air compressors tend to have "
        "large motor-system energy consumption. Individual "
        "plants should review their own processes because "
        "motor requirements vary between facilities, even "
        "within the same industry.",
    ),
    make_question(
        "Q015",
        "Which financial outputs does MotorMaster+ provide "
        "in its life-cycle cost analysis?",
        "financial_analysis",
        "factual",
        [37],
        ["p37_c0"],
        "MotorMaster+ provides rate of return on investment, "
        "levelized cost of energy savings, net present value "
        "and benefit-to-cost ratio.",
    ),
    make_question(
        "Q016",
        "What additional capabilities does MotorMaster "
        "International provide compared with MotorMaster+?",
        "software_tools",
        "comparison",
        [37],
        ["p37_c1"],
        "MotorMaster International supports repair and "
        "replacement analysis for a broader range of motors, "
        "multiple currencies, regional number formats, "
        "efficiency-benefit calculations with kW- and "
        "kVA-based demand charges, and identification of "
        "best-available replacement motors.",
    ),
    {
        "question_id": "Q017",
        "question": "What is the Wi-Fi password for the "
                    "industrial facility described in the manual?",
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

    existing_ids = {
        question["question_id"] for question in questions
    }
    new_ids = {
        question["question_id"] for question in NEW_QUESTIONS
    }
    duplicates = existing_ids & new_ids

    if duplicates:
        raise SystemExit(
            f"Questions already exist: {sorted(duplicates)}. "
            "No changes made."
        )

    updated = questions + NEW_QUESTIONS

    # Validate a temporary copy before replacing the dataset.
    temp_path = DATASET_PATH.with_name(
        "dev_questions.batch1.tmp.json"
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
        sum(question["answerable"] for question in updated),
    )
    print(
        "Unanswerable:",
        sum(not question["answerable"] for question in updated),
    )


if __name__ == "__main__":
    main()