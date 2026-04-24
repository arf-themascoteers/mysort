import csv
from pathlib import Path
import utils
import torch
import torch.nn as nn
import func_generator


class SortModel(nn.Module):
    def __init__(self, array_length):
        super().__init__()
        evenly_spaced = torch.linspace(0, 1, array_length)
        self.indices = nn.Parameter(evenly_spaced)

    def forward(self, array):
        clamped_indices = self.indices.clamp(0, 1)
        f = func_generator.make_piecewise_linear(clamped_indices, array)

        alpha = 10
        delta = 0.0005

        sorted_indices, _ = torch.sort(clamped_indices)
        left_points = sorted_indices[:-1] + delta
        right_points = sorted_indices[1:] - delta

        left_items = f(left_points)
        right_items = f(right_points)

        gaps = torch.relu(left_items - right_items) 
        spacing = gaps * torch.abs(sorted_indices[:-1] - sorted_indices[1:])
        total_loss = gaps.sum() + 0.001 * spacing.sum()
        return alpha * total_loss

    def get_indices(self):
        return torch.argsort(self.indices)

def predict(array, num_epochs=200, lr=0.1, verbose=False):
    array = torch.as_tensor(array, dtype=torch.float32)
    model = SortModel(len(array))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    csv_file = None
    writer = None
    col_width = 12
    if verbose:
        csv_path = Path(__file__).parent / "training_log.csv"
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        header = utils.csv_header(len(array))
        writer.writerow(header)
        print("".join(f"{h:>{col_width}}" for h in header))

    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model(array)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.indices.clamp_(0, 1)
            indices = model.get_indices()
            new_array = array[indices]

            if verbose:
                row = utils.csv_row(epoch, loss.item(), model, new_array)
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
        indices = model.get_indices()
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
        sorted_array = predict(original, verbose=False)
        print(f"original: {original.tolist()}")
        print(f"sorted:   {sorted_array.tolist()}")
        print()


if __name__ == "__main__":
    main()
