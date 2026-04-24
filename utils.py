import torch

def ranks_of(tensor):
    sorted_positions = torch.argsort(tensor)
    ranks = torch.argsort(sorted_positions)
    return ranks

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


def csv_row(epoch, loss_value, model, array):
    score = score_sortedness(array)
    row = [epoch, loss_value, score]
    for value in model.indices.detach().tolist():
        row.append(value)
    for value in array.tolist():
        row.append(value)
    return row