import csv
from pathlib import Path
from array_fn import ArrayFunction
import torch
import torch.nn as nn


def ranks_of(tensor):
    sorted_positions = torch.argsort(tensor)
    ranks = torch.argsort(sorted_positions)
    return ranks


def unsortedness_loss(array):
    alpha = 0.1
    total_out_of_order = torch.zeros(())
    for position in range(1, len(array)):
        left_item = array[position - 1]
        right_item = array[position]
        gap = torch.clamp(left_item - right_item, min=0.0)
        total_out_of_order = total_out_of_order + gap
    return alpha * total_out_of_order


class SortModel(nn.Module):
    def __init__(self, array):
        super().__init__()
        array_length = len(array)
        evenly_spaced = torch.linspace(0, 1, array_length)
        self.indices = nn.Parameter(evenly_spaced)
        self.array_fn = ArrayFunction(array)
        self.reordered = array

    def forward(self):
        clamped_indices = self.indices.clamp(0, 1)
        reordered = self.array_fn(clamped_indices)
        return unsortedness_loss(reordered)
    
    def update_array(self):
        clamped_indices = self.indices.clamp(0, 1)
        ranks = ranks_of(clamped_indices)
        self.reordered = self.reordered[ranks]

    def get_updated_array(self):
        return self.reordered


def score_sortedness(array):
    ideal_ranks = ranks_of(array)
    current_ranks = torch.arange(len(array))
    rank_errors = (current_ranks - ideal_ranks).abs()
    return rank_errors.sum().item()


def csv_header(array_length):
    columns = ["epoch", "loss", "score"]
    for position in range(array_length):
        columns.append(f"idx_{position}")
    for position in range(array_length):
        columns.append(f"reordered_{position}")
    return columns


def csv_row(epoch, loss_value, score_value, model):
    row = [epoch, loss_value, score_value]
    for value in model.indices.detach().tolist():
        row.append(value)
    for value in model.get_updated_array().tolist():
        row.append(value)
    return row


if __name__ == "__main__":
    torch.manual_seed(0)

    array = torch.tensor([0.7, 0.2, 0.9, 0.1, 0.5, 0.3])
    model = SortModel(array)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 200

    csv_path = Path(__file__).parent / "training_log.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header(len(array)))

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            model.update_array()

            with torch.no_grad():
                score = score_sortedness(model.get_updated_array())

            writer.writerow(csv_row(epoch, loss.item(), score, model))

    print(f"training log written to {csv_path}")
