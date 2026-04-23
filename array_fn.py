import torch

class ArrayFunction(torch.nn.Module):
    def __init__(self, y):
        super().__init__()
        y = torch.as_tensor(y, dtype=torch.float32)
        self.register_buffer("y", y)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.y.dtype, device=self.y.device)

        if torch.any((x < 0) | (x > 1)):
            raise ValueError("undefined outside [0, 1]")

        n = self.y.numel()

        if n == 1:
            return torch.full_like(x, self.y[0])

        idx = torch.round(x * (n - 1)).long()
        return self.y[idx]

if __name__ == "__main__":
    f = ArrayFunction([10.0, 20.0, 50.0, 80.0])
    x = torch.tensor([0.0, 0.25, 0.5, 1.0], requires_grad=True)
    x = torch.linspace(0, 1, 4)
    z = f(x)
    print(z)
