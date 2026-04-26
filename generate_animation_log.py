import csv
from pathlib import Path
import torch

from sort_model import SortModel


ARRAY = [0.5, 0.2, 0.7, 0.4, 0.1, 0.6]
OUTPUT_PATH = Path(__file__).parent / "animation_log.csv"


def header_columns(array_length):
    columns = ["epoch", "loss"]
    for position in range(array_length):
        columns.append(f"idx_{position}")
    for position in range(array_length):
        columns.append(f"perm_{position}")
    return columns


def row_for_epoch(epoch, loss_value, model):
    indices_list = model.indices.detach().tolist()
    permutation_list = model.get_indices().tolist()
    row = [epoch, loss_value]
    row.extend(indices_list)
    row.extend(permutation_list)
    return row


def train_and_log():
    torch.manual_seed(0)
    array = torch.tensor(ARRAY, dtype=torch.float32)
    model = SortModel(len(array))
    optimizer = torch.optim.SGD(model.parameters(), lr=model.LEARNING_RATE)

    with open(OUTPUT_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header_columns(len(array)))

        for epoch in range(model.NUM_EPOCHS):
            optimizer.zero_grad()
            loss = model(array)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.indices.clamp_(0, 1)
            writer.writerow(row_for_epoch(epoch, loss.item(), model))
            if loss.item() < model.min_loss:
                break

    print(f"animation log written to {OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_log()
