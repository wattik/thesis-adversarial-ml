import torch
from torch.autograd import grad


def jacobian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    rows = []

    for f_i in f.flatten():
        df_i = grad(f_i, x, retain_graph=True)[0]
        rows.append(df_i)

    return torch.stack(rows, dim=-1)


if __name__ == '__main__':
    a = torch.tensor(
        [
            [6, 2],
            [2, 8]
        ], dtype=torch.float)
    x = torch.tensor([[1, 1]], requires_grad=True, dtype=torch.float)

    f = x.matmul(a)

    print(jacobian(f, x))
    print(jacobian(f, x))
