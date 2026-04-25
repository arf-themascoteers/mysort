import csv
from pathlib import Path
import torch
import torch.nn as nn


class FuncGenerator:
    @staticmethod
    def make_smooth(xp, yp, tau=0.01):
        xp = xp.float()
        yp = yp.float()

        def f(x):
            x = torch.as_tensor(x).float()
            original_shape = x.shape
            x_flat = x.reshape(-1)

            dist_sq = (x_flat.unsqueeze(-1) - xp.unsqueeze(0)) ** 2
            weights = torch.softmax(-dist_sq / tau, dim=-1)
            y = (weights * yp.unsqueeze(0)).sum(dim=-1)

            return y.reshape(original_shape)

        return f

    @staticmethod
    def make_piecewise_linear(xp, yp):
        xp = xp.float()
        yp = yp.float()

        order = torch.argsort(xp.detach())
        xp = xp[order]
        yp = yp[order]

        def f(x):
            x = torch.as_tensor(x).float()
            original_shape = x.shape
            x_flat = x.reshape(-1)

            lo = xp[0].detach()
            hi = xp[-1].detach()
            x_clamped = x_flat.clamp(lo, hi)

            i = torch.searchsorted(xp.detach(), x_clamped.detach(), right=True) - 1
            i = i.clamp(0, len(xp) - 2)

            x0 = xp[i]
            x1 = xp[i + 1]
            y0 = yp[i]
            y1 = yp[i + 1]

            y = y0 + (x_clamped - x0) * (y1 - y0) / (x1 - x0)

            return y.reshape(original_shape)

        return f


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

    def forward(self, array):
        clamped_indices = self.indices.clamp(0, 1)
        lam = 0.1
        m = torch.mean(clamped_indices)
        clamped_indices = (1 - lam) * clamped_indices + lam * m

        eps = 1e-6
        positional = eps * torch.arange(
            clamped_indices.numel(),
            dtype=clamped_indices.dtype,
            device=clamped_indices.device,
        )
        clamped_indices = clamped_indices + positional

        f = FuncGenerator.make_piecewise_linear(clamped_indices, array)

        alpha = 100
        delta = 0.0000005

        sorted_indices, _ = torch.sort(clamped_indices)
        left_points = sorted_indices[:-1] + delta
        right_points = sorted_indices[1:] - delta

        left_items = f(left_points)
        right_items = f(right_points)

        gaps = torch.relu(left_items - right_items)
        gaps = gaps / (gaps.sum() + 0.00001)
        spacing = gaps * torch.abs(sorted_indices[:-1] - sorted_indices[1:])
        total_loss = gaps.sum() + 0.001 * spacing.sum()
        return alpha * total_loss

    def get_indices(self):
        return torch.argsort(self.indices)

    @classmethod
    def predict(cls, array, verbose=False):
        array = torch.as_tensor(array, dtype=torch.float32)
        model = cls(len(array))
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

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
                optimizer.zero_grad()
                loss = model(array)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.indices.clamp_(0, 1)
                indices = model.get_indices()
                new_array = array[indices]

                if verbose:
                    row = Utils.csv_row(epoch, loss.item(), model, new_array)
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
        sorted_array = SortModel.predict(original, verbose=False)
        print(f"original: {original.tolist()}")
        print(f"sorted:   {sorted_array.tolist()}")
        print()


if __name__ == "__main__":
    main()
