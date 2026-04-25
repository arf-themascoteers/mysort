import torch

from generate_test_cases import load_cases
from sort_model import Utils, predict


CASE_ID = 170


def main():
    torch.manual_seed(0)

    by_id = {cid: (ctype, arr) for cid, ctype, arr in load_cases()}
    if CASE_ID not in by_id:
        raise SystemExit(f"case id {CASE_ID} not found in test_cases.csv")
    case_type, array = by_id[CASE_ID]

    original = torch.tensor(array, dtype=torch.float32)
    print(f"id: {CASE_ID}  case_type: {case_type}")
    print(f"original: {original.tolist()}")
    print(f"original score: {Utils.score_sortedness(original)}")
    print()

    result = predict(original.clone(), verbose=True)

    print()
    print(f"sorted:   {result.tolist()}")
    print(f"final score: {Utils.score_sortedness(result)}")


if __name__ == "__main__":
    main()
