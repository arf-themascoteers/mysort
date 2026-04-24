import sort_model
import torch


def format_row(label, values):
    cells = [f"{value:>6}" for value in values]
    return label.ljust(12) + "\t" + "\t".join(cells)


def round_list(tensor, decimals):
    return [round(value, decimals) for value in tensor.tolist()]


def to_int_list(tensor):
    return tensor.tolist()


if __name__ == "__main__":
    tensor = torch.tensor([0,2,4,1], dtype=torch.float32)
    ranks = sort_model.ranks_of(tensor)
    argsorted = torch.argsort(tensor)
    sorted_values, _ = torch.sort(tensor)

    positions = list(range(len(tensor)))

    print(format_row("position", positions))
    print(format_row("tensor", round_list(tensor, 2)))
    print(format_row("sorted", round_list(sorted_values, 2)))
    print(format_row("argsort", to_int_list(argsorted)))
    print(format_row("ranks_of", to_int_list(ranks)))
