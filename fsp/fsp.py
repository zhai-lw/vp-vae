import torch
from torch import nn

from ..vp import l_utils
from .cdf_act import build_act


class FSP(nn.Module):
    def __init__(self, levels: list[int], act_name: str = 'tanh', vector_norm: str = 'none',
                 eta: float = 1., quantize_rate: float = 0.,
                 need_inv_act: bool = False, ):
        super().__init__()
        self.levels = nn.Buffer(torch.tensor(levels, dtype=torch.int32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.basis = nn.Buffer(basis, persistent=False)

        self.codebook_dim: int = self.levels.numel()
        self.codebook_size: int = torch.prod(self.levels).item()

        self.vector_norm = l_utils.VectorNorm.build(vector_norm, dim=self.codebook_dim)

        self.act_name = act_name
        self.act_func, self.inv_act_func = build_act(self.act_name)
        self.need_inv_act = need_inv_act

        self.eta = eta
        self.quantize_rate = quantize_rate

    def __repr__(self):
        return (f"FSP("
                f"levels={self.levels}, "
                f"codebook_size={self.codebook_size}, "
                f"act_name={self.act_name}, "
                f"need_inv_act={self.need_inv_act}, "
                f"eta={self.eta}, "
                f"quantize_rate={self.quantize_rate}, "
                f")")

    def quantize_act_value(self, act_z, eps: float):
        """
        Args:
            act_z: [0, 1]
            eps: 1e-6
        Returns:
            q_act_z: [0, 1]
            level_indices: {0, 1, 2, ..., (l-1)}
        """
        level_indices = (act_z.clamp_max(1 - eps) * self.levels).floor()
        q_act_z = (level_indices + 0.5) / self.levels

        q_act_z = act_z + (q_act_z - act_z).detach()
        return q_act_z, level_indices.detach()

    def forward(self, features: torch.Tensor, eps: float = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        eps = eps or torch.finfo(features.dtype).eps
        feature_shape = features.shape
        z = features.view(-1, feature_shape[-1])
        z, norm_loss, norm_info = self.vector_norm(z)

        act_z = self.act_func(z)
        q_act_z, level_indices = self.quantize_act_value(act_z, eps=eps)
        other_info = {}

        quantize_rate = self.quantize_rate if self.training else 1.0
        if quantize_rate < 1.:
            p_max_norm = self.eta / (self.levels * 2)
            perturbations = p_max_norm * (torch.rand_like(act_z) * 2 - 1)
            proposal = act_z + perturbations
            accept_mask = (proposal > 0.0) & (proposal < 1.0)
            other_info['p_accept_prob'] = accept_mask.float().mean()
            p_act_z = torch.where(accept_mask, proposal, act_z)

            p_mask = torch.rand_like(q_act_z) > quantize_rate
            q_act_z[p_mask] = p_act_z[p_mask]

        if self.need_inv_act:
            q_z = self.inv_act_func(q_act_z.clamp(eps, 1 - eps))
            q_z = z + (q_z - z).detach()
        else:
            # q_z = q_act_z * 2 - 1
            q_z = (q_act_z - 0.5) / 0.28867513459481287  # to make q_z.var() -> 1.0
        indices = self.level_indices_to_indices(level_indices)

        return (q_z.reshape(feature_shape),
                indices.reshape(feature_shape[:-1]),
                {
                    "level_indices": level_indices.reshape(feature_shape),
                    'norm_loss': norm_loss,
                    'norm_info': norm_info,
                    **other_info,
                })

    def level_indices_to_indices(self, level_indices):
        return (level_indices * self.basis).sum(dim=-1).to(torch.int32)

    def indices_to_level_indices(self, indices):
        return (indices.unsqueeze(-1) // self.basis) % self.levels

    def indices_to_act_value(self, indices: torch.Tensor):
        level_indices = self.indices_to_level_indices(indices)
        return (level_indices + 0.5) / self.levels

    def indices_to_codes(self, indices: torch.Tensor):
        return self.inv_act_func(self.indices_to_act_value(indices))
