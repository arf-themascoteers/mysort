import csv
import json
import random
from pathlib import Path


CASES_CSV = Path(__file__).parent / "test_cases.csv"
HEADER = ["id", "case_type", "array"]


def build_test_cases():
    cases = []
    rng = random.Random(0)
    lengths = [3, 4, 5, 6, 8, 10, 12, 16, 20]

    for length in lengths:
        cases.append(("ascending", [(i + 1) / (length + 1) for i in range(length)]))
        cases.append(("descending", [(length - i) / (length + 1) for i in range(length)]))
        cases.append(("constant", [0.5] * length))
        cases.append(("two_values", [0.2 if i % 2 == 0 else 0.8 for i in range(length)]))

        nearly = [(i + 1) / (length + 1) for i in range(length)]
        if length >= 2:
            nearly[0], nearly[1] = nearly[1], nearly[0]
        cases.append(("nearly_sorted", nearly))

        reversed_pair = [(i + 1) / (length + 1) for i in range(length)]
        if length >= 2:
            reversed_pair[-1], reversed_pair[-2] = reversed_pair[-2], reversed_pair[-1]
        cases.append(("last_pair_swapped", reversed_pair))

        for trial in range(10):
            cases.append((f"random_seed_{trial}", [rng.random() for _ in range(length)]))

        for trial in range(3):
            base = rng.random()
            arr = [min(1.0, max(0.0, base + rng.gauss(0, 0.02))) for _ in range(length)]
            cases.append((f"clustered_{trial}", arr))

    return cases


def write_cases(path=CASES_CSV):
    cases = build_test_cases()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i, (case_type, arr) in enumerate(cases):
            writer.writerow([i, case_type, json.dumps(arr)])
    return len(cases)


def load_cases(path=CASES_CSV):
    cases = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            case_id = int(row[0])
            case_type = row[1]
            array = json.loads(row[2])
            cases.append((case_id, case_type, array))
    return cases


def main():
    n = write_cases()
    print(f"wrote {n} cases to {CASES_CSV}")


if __name__ == "__main__":
    main()
