import csv
import random
import time
from pathlib import Path

import torch

from sort_model import Utils, predict


def is_sorted(values):
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def build_test_cases():
    cases = []
    rng = random.Random(0)

    lengths = [3, 4, 5, 6, 8, 10, 12, 16, 20]

    for length in lengths:
        cases.append(("ascending", list((i + 1) / (length + 1) for i in range(length))))
        cases.append(("descending", list((length - i) / (length + 1) for i in range(length))))
        cases.append(("constant", [0.5] * length))
        two_vals = [0.2 if i % 2 == 0 else 0.8 for i in range(length)]
        cases.append(("two_values", two_vals))
        nearly = [(i + 1) / (length + 1) for i in range(length)]
        if length >= 2:
            nearly[0], nearly[1] = nearly[1], nearly[0]
        cases.append(("nearly_sorted", nearly))
        reversed_pair = [(i + 1) / (length + 1) for i in range(length)]
        if length >= 2:
            reversed_pair[-1], reversed_pair[-2] = reversed_pair[-2], reversed_pair[-1]
        cases.append(("last_pair_swapped", reversed_pair))

        for trial in range(10):
            arr = [rng.random() for _ in range(length)]
            cases.append((f"random_seed_{trial}", arr))

        for trial in range(3):
            base = rng.random()
            arr = [min(1.0, max(0.0, base + rng.gauss(0, 0.02))) for _ in range(length)]
            cases.append((f"clustered_{trial}", arr))

    return cases


def run_tests(output_csv: Path):
    torch.manual_seed(0)
    cases = build_test_cases()

    rows = []
    columns = [
        "test_id",
        "length",
        "case_type",
        "original_array",
        "sorted_array",
        "original_score",
        "final_score",
        "perfectly_sorted",
        "monotonic",
        "elapsed_seconds",
    ]

    print(
        f"{'#':>4} {'len':>4} {'case_type':<22} {'orig_score':>10} "
        f"{'final':>6} {'result':<8} {'array':<60}"
    )
    print("-" * 120)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        f.flush()

        for test_id, (case_type, array) in enumerate(cases):
            original = torch.tensor(array, dtype=torch.float32)
            original_score = Utils.score_sortedness(original)

            start = time.perf_counter()
            result = predict(original.clone(), verbose=False)
            elapsed = time.perf_counter() - start

            final_score = Utils.score_sortedness(result)
            result_list = result.tolist()
            monotonic = is_sorted(result_list)
            perfectly_sorted = final_score == 0

            status = "PASS" if perfectly_sorted else "FAIL"
            array_preview = ", ".join(f"{v:.3f}" for v in array)
            if len(array_preview) > 58:
                array_preview = array_preview[:55] + "..."
            print(
                f"{test_id:>4} {len(array):>4} {case_type:<22} {int(original_score):>10} "
                f"{int(final_score):>6} {status:<8} [{array_preview}]"
            )

            row = [
                test_id,
                len(array),
                case_type,
                [round(v, 6) for v in array],
                [round(v, 6) for v in result_list],
                int(original_score),
                int(final_score),
                int(perfectly_sorted),
                int(monotonic),
                round(elapsed, 4),
            ]
            writer.writerow(row)
            f.flush()
            rows.append(row)

    return columns, rows


def summarize(columns, rows):
    total = len(rows)
    if total == 0:
        return "No tests were run."

    idx = {name: i for i, name in enumerate(columns)}
    perfect = sum(1 for r in rows if r[idx["perfectly_sorted"]] == 1)
    monotonic = sum(1 for r in rows if r[idx["monotonic"]] == 1)
    avg_final = sum(r[idx["final_score"]] for r in rows) / total
    avg_original = sum(r[idx["original_score"]] for r in rows) / total
    total_time = sum(r[idx["elapsed_seconds"]] for r in rows)
    worst = max(rows, key=lambda r: r[idx["final_score"]])

    summary = (
        f"Ran {total} tests: {perfect} perfectly sorted ({perfect / total:.1%}), "
        f"{monotonic} monotonic ({monotonic / total:.1%}); "
        f"avg score {avg_original:.2f} -> {avg_final:.2f}; "
        f"worst case {worst[idx['case_type']]} (len={worst[idx['length']]}, "
        f"final_score={worst[idx['final_score']]}); "
        f"total time {total_time:.2f}s."
    )
    return summary


def main():
    output_csv = Path(__file__).parent / "test_results.csv"
    columns, rows = run_tests(output_csv)
    summary = summarize(columns, rows)
    print(f"results written to {output_csv}")
    print(summary)


if __name__ == "__main__":
    main()
