import torch

from sort_model import Utils, predict


def main():
    torch.manual_seed(0)

    array = [0.783799, 0.303313, 0.476597]

    original = torch.tensor(array, dtype=torch.float32)
    print(f"original: {original.tolist()}")
    print(f"original score: {Utils.score_sortedness(original)}")
    print()

    result = predict(original.clone(), verbose=True)

    print()
    print(f"sorted:   {result.tolist()}")
    print(f"final score: {Utils.score_sortedness(result)}")


if __name__ == "__main__":
    main()
