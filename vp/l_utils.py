import torch


def batch_stats(batch: torch.Tensor, eps: float = 1e-8):
    variance, mean = torch.var_mean(batch, dim=0, unbiased=True)
    std = variance.sqrt().clamp_min(eps)
    z_scores = (batch - mean) / std
    skewness = torch.mean(z_scores.pow(3), dim=0)
    kurtoses = torch.mean(z_scores.pow(4), dim=0) - 3.0
    return mean, variance, skewness, kurtoses


class ZeroMeanBatchNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.running_var = torch.nn.Buffer(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        :param x: shape (N, C)
        :return: shape (N, C)
        """
        if self.training:
            batch_var = torch.mean(x ** 2, dim=0)  # ! detach or not
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_normalized = x / torch.sqrt(batch_var + self.eps)
        else:
            x_normalized = x / torch.sqrt(self.running_var + self.eps)

        return x_normalized


class PassiveNormalization(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.01):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.nn.Buffer(torch.zeros(dim))
        self.running_var = torch.nn.Buffer(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        :param x: shape (N, C)
        :return: shape (N, C)
        """
        if self.training:
            batch_var, batch_mean = torch.var_mean(x, dim=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()

            # norm_mean, norm_var = batch_mean, batch_var   # batch_norm
            # norm_mean, norm_var = batch_mean.detach(), batch_var.detach()  # bn_detach
            norm_mean, norm_var = self.running_mean, self.running_var  # pass_norm
            # norm_mean, norm_var = ((batch_mean + self.running_mean) / 2, (batch_var + self.running_var) / 2)  # half_n
        else:
            norm_mean, norm_var = self.running_mean, self.running_var

        x_normalized = (x - norm_mean) / torch.sqrt(norm_var + self.eps)
        return x_normalized


class VectorNorm(torch.nn.Module):
    def __init__(self, norm_func: str = 'none', z_dim: int = 0,
                 l1_target: float = 0.0, l1_weight: float = 0.1,
                 l2_target: float = 1.0, l2_weight: float = 0.07,
                 l3_target: float = 0.0, l3_weight: float = 0.06,
                 l4_target: float = 0.0, l4_weight: float = 0.05,
                 eps: float = 1e-8):
        super(VectorNorm, self).__init__()

        if norm_func == 'none':
            self.norm_layer = torch.nn.Identity()
        elif norm_func == 'batch_norm':
            self.norm_layer = torch.nn.BatchNorm1d(z_dim, affine=False)
        elif norm_func == 'batch_norm_notrack':
            self.norm_layer = torch.nn.BatchNorm1d(z_dim, affine=False, track_running_stats=False)
        elif norm_func == 'var_norm':
            self.norm_layer = ZeroMeanBatchNorm(z_dim)
        elif norm_func == 'passive_norm':
            self.norm_layer = PassiveNormalization(z_dim)
        else:
            raise NotImplementedError(f"Normalization function '{norm_func}' is not implemented.")

        self.l1_target, self.l1_weight = l1_target, l1_weight
        self.l2_target, self.l2_weight = l2_target, l2_weight
        self.l3_target, self.l3_weight = l3_target, l3_weight
        self.l4_target, self.l4_weight = l4_target, l4_weight
        self.eps = eps

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        mean, variance, skewness, kurtoses = batch_stats(z, self.eps)
        norm_z = self.norm_layer(z)
        norm_loss = (((mean - self.l1_target) ** 2).mean() * self.l1_weight
                     + ((variance - self.l2_target) ** 2).mean() * self.l2_weight
                     + ((skewness - self.l3_target) ** 2).mean() * self.l3_weight
                     + ((kurtoses - self.l4_target) ** 2).mean() * self.l4_weight)
        return norm_z, norm_loss, {
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtoses': kurtoses,
        }

    @classmethod
    def build(cls, vn_name: str = 'none', dim: int = 0):
        match vn_name:
            case 'none':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_weight=0., l2_weight=0., l3_weight=0., l4_weight=0.)
            case 'none-var':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_target=0.0, l1_weight=0.1,
                    l2_target=1.0, l2_weight=0.07,
                    l3_target=0.0, l3_weight=0.,
                    l4_target=0.0, l4_weight=0.
                )
            case 'none-kurt':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_target=0.0, l1_weight=0.1,
                    l2_target=1.0, l2_weight=0.07,
                    l3_target=0.0, l3_weight=0.06,
                    l4_target=0.0, l4_weight=0.05
                )
            case 'none-var_tanh':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_target=0.0, l1_weight=0.1,
                    l2_target=0.8225, l2_weight=0.07,
                    l3_target=0.0, l3_weight=0.,
                    l4_target=0.0, l4_weight=0.
                )
            case 'none-var_sigmoid':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_target=0.0, l1_weight=0.1,
                    l2_target=3.29, l2_weight=0.07,
                    l3_target=0.0, l3_weight=0.,
                    l4_target=0.0, l4_weight=0.
                )
            case 'none-var_laplace':
                return cls(
                    norm_func='none', z_dim=dim,
                    l1_target=0.0, l1_weight=0.1,
                    l2_target=2.0, l2_weight=0.07,
                    l3_target=0.0, l3_weight=0.,
                    l4_target=0.0, l4_weight=0.
                )
            case _:
                raise ValueError(f'Unknown vector_norm: {vn_name}')


def batch_stats_test():
    from torch.distributions import Laplace
    dist = Laplace(0.5, 1.0)
    samples = dist.sample(sample_shape=(10000, 4))
    mean, variance, skewness, kurtoses = batch_stats(samples)
    print("Mean:", mean)
    print("Variance:", variance)
    print("Skewness:", skewness)
    print("Kurtoses:", kurtoses)


if __name__ == '__main__':
    batch_stats_test()
