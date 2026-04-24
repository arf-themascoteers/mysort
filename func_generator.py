import torch

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

if __name__ == "__main__":
    xp = torch.tensor([0.0, 0.25, 0.8, 0.7])
    yp = torch.tensor([0, 2, 4, 1])
    f = make_piecewise_linear(xp, yp)

    x=torch.tensor(0.5, requires_grad=True)
    y = f(x)
    y.backward()
    print(x.grad)