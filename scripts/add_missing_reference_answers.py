import json
import shutil
from pathlib import Path

from app.eval_dataset import load_evaluation_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "dev_questions.json"

BACKUP_PATH = (
    ROOT / "data" / "index" /
    "dev_questions.before_reference_answers.json"
)

REFERENCE_ANSWERS = {
    "Q001": (
        "Increasing motor operating speed can increase the load "
        "imposed by rotating equipment and therefore increase "
        "energy consumption. In the sourcebook's example, "
        "increasing speed from 1,760 to 1,770 RPM can increase "
        "the load by up to 1.6%, while a 20-RPM increase can "
        "raise both load and energy consumption by 3.3%. "
        "These are illustrative figures, not universal values "
        "for every motor-driven system."
    ),

    "Q002": (
        "For many EPAct Design A and B motors, efficiency is "
        "relatively constant around 70% to 80% of rated load "
        "and may drop slightly at full load. Efficiency may "
        "begin to decline below 50% of rated load and declines "
        "dramatically below 40%. Slight oversizing may improve "
        "efficiency, but substantial oversizing can reduce it."
    ),

    "Q003": (
        "Performance-improvement opportunities often arise "
        "from motors that are poorly sized, improperly "
        "configured, or inadequately maintained. In industrial "
        "facilities, much of the motor-system energy consumption "
        "may be concentrated in a few large motors that operate "
        "for long periods. These energy-intensive systems are "
        "important candidates for improvements, even when "
        "production-related downtime makes modifications "
        "difficult."
    ),

    "Q004": (
        "In a series DC motor, the field and armature are "
        "connected in series, so the same current passes "
        "through both. Before magnetic saturation, torque "
        "increases in proportion to the square of current. "
        "Beyond saturation, increases in load are directly "
        "proportional to increases in current."
    ),

    "Q005": (
        "Breakdown torque is the maximum torque a motor can "
        "produce without an abrupt drop in speed. If the load "
        "exceeds breakdown torque, the motor stalls and can "
        "overheat rapidly, potentially causing insulation "
        "failure if it is not properly protected."
    ),

    "Q006": (
        "A formal motor management program supports proactive, "
        "cost-effective management of motor systems instead of "
        "treating them as overlooked assets. It can establish "
        "motor repair-versus-replacement and purchasing policies, "
        "maintain a motor inventory, track motor life, create "
        "a spare-motor inventory, and schedule required "
        "maintenance."
    ),

    "Q007": (
        "Motor and pump alignment problems can result from "
        "poor installation, foundation movement, or bearing "
        "system wear. During installation, force-fitting piping "
        "that does not align with pump flanges can pull the "
        "motor-pump shafts out of alignment. Welding can also "
        "distort foundations and affect machinery alignment."
    ),

    "Q008": (
        "A long motor feeder cable can introduce enough "
        "inductance to create resonance in the drive-cable-motor "
        "circuit. Reflected voltage waves can then produce "
        "voltage overshoot at the motor terminals, potentially "
        "exceeding twice the normal voltage. A large part of "
        "the overshoot can appear across the first turn of "
        "the stator winding, stressing its turn-to-turn "
        "insulation and causing electrical discharge."
    ),

    "Q009": (
        "When a project produces a consistent annual benefit, "
        "simple payback is calculated by dividing the initial "
        "investment by the annual benefit. It ignores the "
        "time value of money, treating money received today "
        "and money received in the future as equivalent."
    ),

    "Q010": (
        "According to the sourcebook, the DOE Advanced "
        "Manufacturing Office assists industry through "
        "partnerships that develop energy-efficient technologies "
        "and processes, technical assistance for assessing "
        "energy intensity and identifying efficiency "
        "opportunities, and training and information resources. "
        "Its activities address energy and cost savings, "
        "waste reduction, pollution prevention, and improved "
        "environmental performance. The sourcebook also "
        "identifies no-cost energy assessments available "
        "to qualifying small and medium-sized manufacturers "
        "through Industrial Assessment Centers."
    ),
}


def main():
    questions = load_evaluation_dataset(DATASET_PATH)

    expected_ids = set(REFERENCE_ANSWERS)

    missing_ids = {
        question["question_id"]
        for question in questions
        if question["answerable"]
        and not question.get("reference_answer")
    }

    if missing_ids != expected_ids:
        raise SystemExit(
            "Unexpected missing-answer IDs.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Actual: {sorted(missing_ids)}\n"
            "No changes made."
        )

    if len(questions) != 42:
        raise SystemExit(
            f"Expected 42 questions, found {len(questions)}. "
            "No changes made."
        )

    for question in questions:
        question_id = question["question_id"]

        if question_id in REFERENCE_ANSWERS:
            question["reference_answer"] = (
                REFERENCE_ANSWERS[question_id]
            )

    temp_path = DATASET_PATH.with_name(
        "dev_questions.references.tmp.json"
    )

    temp_path.write_text(
        json.dumps(
            questions,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Validate the complete updated dataset before replacing it.
    validated = load_evaluation_dataset(temp_path)

    still_missing = [
        question["question_id"]
        for question in validated
        if question["answerable"]
        and not question.get("reference_answer")
    ]

    if still_missing:
        temp_path.unlink()
        raise SystemExit(
            f"Missing answers remain: {still_missing}"
        )

    # Keep a local backup outside the tracked evaluation directory.
    BACKUP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(DATASET_PATH, BACKUP_PATH)

    # Replace only after all checks have passed.
    temp_path.replace(DATASET_PATH)

    print("Reference answers added: 10")
    print("Total development questions:", len(validated))
    print(
        "Answerable:",
        sum(q["answerable"] for q in validated),
    )
    print(
        "Missing reference answers:",
        len(still_missing),
    )
    print("Local backup:", BACKUP_PATH)


if __name__ == "__main__":
    main()