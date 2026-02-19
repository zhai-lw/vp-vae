import math

import torch


def build_act(act_name: str):
    """
    Args:
        act_name:
    Returns:
        act_func: (-∞, +∞) -> [0, 1]
        inv_act_func: (0, 1) -> (-∞, +∞)
    """
    match act_name:
        case "tanh":
            act_func, inv_act_func = tanh_act, tanh_inv_act
        case "sigmoid":
            act_func, inv_act_func = sigmoid_act, sigmoid_inv_act
        case "normal":
            act_func, inv_act_func = normal_act, normal_inv_act
        case "laplace":
            act_func, inv_act_func = laplace_act, laplace_inv_act
        case "cauchy":
            act_func, inv_act_func = cauchy_act, cauchy_inv_act
        case _:
            raise NotImplementedError(f"act_func({act_name}) has not been implemented yet.")
    return act_func, inv_act_func


def tanh_act(z: torch.Tensor) -> torch.Tensor:
    # (-∞, +∞) -> [0, 1]
    return (torch.tanh(z) + 1) / 2


def tanh_inv_act(p: torch.Tensor) -> torch.Tensor:
    # (0, 1) -> (-∞, +∞)
    return torch.arctanh(p * 2 - 1)


def sigmoid_act(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def sigmoid_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.logit(p)


Sqrt2 = math.sqrt(2)


def normal_act(z: torch.Tensor) -> torch.Tensor:
    return (1 + torch.erf(z / Sqrt2)) / 2


def normal_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(2 * p - 1) * Sqrt2


def laplace_act(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.sign(z) * (1 - torch.exp(-torch.abs(z))))


def laplace_inv_act(p: torch.Tensor) -> torch.Tensor:
    return -torch.sign(p - 0.5) * torch.log(1 - 2 * torch.abs(p - 0.5))


def cauchy_act(z: torch.Tensor) -> torch.Tensor:
    return torch.arctan(z) / torch.pi + 1 / 2


def cauchy_inv_act(p: torch.Tensor) -> torch.Tensor:
    return torch.tan((p - 0.5) * torch.pi)


def act_test():
    x = torch.randn(64, 100)
    for act_name in ('tanh', 'sigmoid', 'normal', 'laplace', 'cauchy'):
        act_func, inv_act_func = build_act(act_name)

        y = act_func(x)
        x_hat = inv_act_func(y)

        assert (y > 0.).all() and (y < 1.0).all()

        print(f"{act_name} error: {(x_hat - x).sum()}")


if __name__ == '__main__':
    act_test()
