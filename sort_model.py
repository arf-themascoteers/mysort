import csv
from pathlib import Path

import torch
import torch.nn as nn


def ranks_of(tensor):
    sorted_order = torch.argsort(tensor)
    ranks = torch.empty_like(sorted_order)
    for rank, original_position in enumerate(sorted_order):
        ranks[original_position.item()] = rank
    return ranks


class SortModel(nn.Module):
    def __init__(self, array_length):
        super().__init__()
        evenly_spaced = torch.linspace(0, 1, array_length)
        self.indices = nn.Parameter(evenly_spaced)

    def reorder(self, array):
        clamped_indices = self.indices.clamp(0, 1)
        natural_order = torch.argsort(clamped_indices)
        return array[natural_order]

    def forward(self, array):
        reordered_array = self.reorder(array)
        alpha = 0.1
        loss = torch.tensor(0.0)
        for position in range(1, len(reordered_array)):
            left_item = reordered_array[position - 1]
            this_item = reordered_array[position]
            gap = left_item - this_item
            if gap <= 0:
                continue
            loss = loss + gap * alpha
        return loss


def score_sortedness(array, indices):
    clamped_indices = indices.clamp(0, 1)
    ideal_ranks = ranks_of(array)
    predicted_ranks = ranks_of(clamped_indices)
    differences = (predicted_ranks - ideal_ranks).abs()
    return differences.sum().item()


def build_csv_header(array_length):
    header = ["epoch", "loss", "score"]
    for position in range(array_length):
        header.append(f"idx_{position}")
    for position in range(array_length):
        header.append(f"reordered_{position}")
    return header


def build_csv_row(epoch, loss_value, score_value, indices_tensor, reordered_tensor):
    row = [epoch, loss_value, score_value]
    for value in indices_tensor.tolist():
        row.append(value)
    for value in reordered_tensor.tolist():
        row.append(value)
    return row


if __name__ == "__main__":
    torch.manual_seed(0)

    array = torch.tensor([0.7, 0.2, 0.9, 0.1, 0.5, 0.3])
    model = SortModel(len(array))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 20

    csv_path = Path(__file__).parent / "training_log.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_csv_header(len(array)))

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model(array)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reordered_array = model.reorder(array)
                score = score_sortedness(array, model.indices)

            row = build_csv_row(
                epoch,
                loss.item(),
                score,
                model.indices.detach(),
                reordered_array,
            )
            writer.writerow(row)

    print(f"training log written to {csv_path}")
