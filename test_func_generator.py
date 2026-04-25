import torch

from sort_model import FuncGenerator


def main():
    xp = torch.tensor(
        [
            0.0439, 0.0887, 0.1325, 0.1539, 0.1754, 0.2834, 0.3473, 0.3647,
            0.4299, 0.4875, 0.5264, 0.5608, 0.7052, 0.8286, 0.9439, 0.9439,
        ]
    )
    yp = torch.tensor(
        [
            0.3302, 0.0064, 0.0412, 0.7420, 0.6690, 0.6209, 0.2267, 0.7516,
            0.9997, 0.8731, 0.1055, 0.6997, 0.7271, 0.4609, 0.2879, 0.1683,
        ]
    )

    f = FuncGenerator.make_piecewise_linear(xp, yp)

    print("xp:", xp.tolist())
    print("yp:", yp.tolist())
    print()

    print("1) at the knot points (should equal yp):")
    print("  ", f(xp).tolist())
    print()

    print("2) outside [min, max] (should clamp to endpoints):")
    print("   below lo:", f(torch.tensor([-0.5])).tolist())
    print("   above hi:", f(torch.tensor([1.5])).tolist())
    print()

    print("3) midpoints (linear interp):")
    mids = (xp[:-1] + xp[1:]) / 2
    print("  ", f(mids).tolist())
    print()

    print("4) danger zone around duplicate xp (0.9439, 0.9439):")
    probes = torch.tensor([0.9438, 0.9439, 0.94391, 0.9440])
    out = f(probes)
    print("   x:", probes.tolist())
    print("   y:", out.tolist())
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("   >>> NaN/Inf detected from zero-width segment")
    print()

    print("5) gradient at x=0.5:")
    x = torch.tensor(0.5, requires_grad=True)
    f(x).backward()
    print("   grad:", x.grad.item())


if __name__ == "__main__":
    main()
