import csv
from pathlib import Path
import torch
import torch.nn as nn


class Utils:
    #For reporting
    @staticmethod
    def ranks_of(tensor):
        sorted_positions = torch.argsort(tensor)
        ranks = torch.argsort(sorted_positions)
        return ranks

    # For reporting
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


class TauSort(nn.Module):
    def __init__(self, array_length):
        super().__init__()
        evenly_spaced = torch.linspace(0, 1, array_length)
        self.indices = nn.Parameter(evenly_spaced)
        self.epoch = 0
        self.min_loss = 0.00000001
        self.NUM_EPOCHS = 5000
        self.LEARNING_RATE = 0.02


    def forward(self, array):
        arr_diff = array.unsqueeze(0) - array.unsqueeze(1)
        idx_diff = self.indices.unsqueeze(0) - self.indices.unsqueeze(1)
        raw = -arr_diff * idx_diff
        violations = torch.relu(raw)
        scaled = torch.where(violations > 0, violations + 0.01, violations)
        masked_spacing = torch.where(violations > 0, idx_diff.abs(), torch.zeros_like(idx_diff))
        relevant_spacing = scaled * masked_spacing
        loss_mat = torch.triu(relevant_spacing, diagonal=1)
        return loss_mat.sum()

    def get_indices(self):
        #For log purposes only
        return torch.argsort(self.indices)

    def predict(self, array, verbose=False):
        array = torch.as_tensor(array, dtype=torch.float32)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.LEARNING_RATE)

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
            for epoch in range(self.NUM_EPOCHS):
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
                            formatted.append(f"{value:>{col_width}.8f}")
                        else:
                            formatted.append(f"{value:>{col_width}}")
                    print("".join(formatted))

                if loss.item() < self.min_loss:
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
        model = TauSort(len(original))
        sorted_array = model.predict(original, verbose=False)
        print(f"original: {original.tolist()}")
        print(f"sorted:   {sorted_array.tolist()}")
        print()


if __name__ == "__main__":
    main()
