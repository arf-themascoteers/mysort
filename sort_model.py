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

        gaps = torch.clamp(left_items - right_items, min=0.0)
        return alpha * gaps.sum()

    def get_indices(self):
        return utils.ranks_of(self.indices)

if __name__ == "__main__":
    torch.manual_seed(0)
    array = torch.tensor([0.7, 0.2, 0.9])
    model = SortModel(len(array))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 200

    csv_path = Path(__file__).parent / "training_log.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = utils.csv_header(len(array))
        writer.writerow(header)

        col_width = 12
        print("".join(f"{h:>{col_width}}" for h in header))

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model(array)
            loss.backward()
            optimizer.step()
            indices = model.get_indices()
            new_array = array[indices]
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

    print(f"training log written to {csv_path}")
