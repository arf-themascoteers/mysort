import csv
from pathlib import Path
import torch
import torch.nn as nn


class Utils:
    @staticmethod
    def ranks_of(tensor):
        sorted_positions = torch.argsort(tensor)
        ranks = torch.argsort(sorted_positions)
        return ranks

    @staticmethod
    def score_sortedness(array):
        ideal_ranks = Utils.ranks_of(array)
        current_ranks = torch.arange(len(array))
        rank_errors = (current_ranks - ideal_ranks).abs()
        return rank_errors.sum().item()

    @staticmethod
    def csv_header(array_length):
        columns = ["epoch", "loss", "score"]
        for position in range(array_length):
            columns.append(f"idx_{position}")
        for position in range(array_length):
            columns.append(f"reordered_{position}")
        return columns

    @staticmethod
    def csv_row(epoch, loss_value, model, array):
        score = Utils.score_sortedness(array)
        row = [epoch, loss_value, score]
        for value in model.indices.detach().tolist():
            row.append(value)
        for value in array.tolist():
            row.append(value)
        return row


NUM_EPOCHS = 10000
LEARNING_RATE = 0.1


class SortModel(nn.Module):
    def __init__(self, array_length):
        super().__init__()
        evenly_spaced = torch.linspace(0, 1, array_length)
        self.indices = nn.Parameter(evenly_spaced)
        self.epoch = 0

    def forward(self, array):
        sorted_indices, perm = torch.sort(self.indices)
        sorted_array = array[perm]

        diffs = sorted_array[1:] - sorted_array[:-1]
        violations = torch.relu(-diffs)
        spacing = sorted_indices[1:] - sorted_indices[:-1]
        relevant_spacing = violations * spacing
        #relevant_spacing = relevant_spacing / (relevant_spacing.sum() + 0.00001)
        return (violations + relevant_spacing).sum()

    def get_indices(self):
        return torch.argsort(self.indices)

    def predict(self, array, verbose=False):
        array = torch.as_tensor(array, dtype=torch.float32)
        optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)

        csv_file = None
        writer = None
        col_width = 12
        if verbose:
            csv_path = Path(__file__).parent / "training_log.csv"
            csv_file = open(csv_path, "w", newline="")
            writer = csv.writer(csv_file)
            header = Utils.csv_header(len(array))
            writer.writerow(header)
            print("".join(f"{h:>{col_width}}" for h in header))

        try:
            for epoch in range(NUM_EPOCHS):
                self.epoch = epoch
                optimizer.zero_grad()
                loss = self(array)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self.indices.clamp_(0, 1)
                indices = self.get_indices()
                new_array = array[indices]

                if verbose:
                    row = Utils.csv_row(epoch, loss.item(), self, new_array)
                    writer.writerow(row)
                    formatted = []
                    for value in row:
                        if isinstance(value, float):
                            formatted.append(f"{value:>{col_width}.4f}")
                        else:
                            formatted.append(f"{value:>{col_width}}")
                    print("".join(formatted))

                if loss.item() < 0.000001:
                    break
        finally:
            if csv_file is not None:
                csv_file.close()
                print(f"training log written to {csv_path}")

        with torch.no_grad():
            indices = self.get_indices()
            return array[indices]


def main():
    torch.manual_seed(0)
    test_arrays = [
        [0.7, 0.2, 0.9],
        [0.5, 0.1, 0.8, 0.3],
        [0.4, 0.9, 0.1, 0.7, 0.2, 0.6],
        [0.6, 0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4],
    ]

    for array in test_arrays:
        original = torch.tensor(array, dtype=torch.float32)
        model = SortModel(len(original))
        sorted_array = model.predict(original, verbose=False)
        print(f"original: {original.tolist()}")
        print(f"sorted:   {sorted_array.tolist()}")
        print()


if __name__ == "__main__":
    main()
