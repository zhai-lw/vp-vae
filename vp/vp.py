import logging

import torch
from torch import nn

from . import v_utils, l_utils

log = logging.getLogger("VPVAE")


class VectorPerturbation(nn.Module):
    def __init__(self, codebook_dim: int, codebook_size: int = 1024,
                 eta: float = 1.0, cluster_size_rate: float = 16.0, k4nn: int = 8,
                 vector_norm: str = 'none-var',
                 chunk_size: int = 0):
        super().__init__()
        self.codebook_dim: int = codebook_dim
        self.codebook_size: int = codebook_size
        self.eta: float = eta
        self.cluster_size_rate: float = cluster_size_rate
        self.k4nn: int = k4nn
        self.sample_num_per_cluster: int = round(codebook_dim * cluster_size_rate)

        self.vector_norm = l_utils.VectorNorm.build(vector_norm, dim=self.codebook_dim)

        self.chunk_size = int(16 * 1024 * 256 / self.codebook_size) if chunk_size == 0 else chunk_size
        self.samples_queue = v_utils.SamplesQueue(
            torch.randn(codebook_size * self.sample_num_per_cluster, codebook_dim),
        )
        self.codebook_compiler = v_utils.SamplesKMeans(
            n_clusters=self.codebook_size, max_iter=200, tol=1e-3,
            batch_size=16 * 1024,
        )
        self._compiled = True
        self.codebook = torch.nn.Buffer(torch.randn(codebook_size, self.codebook_dim,
                                                    dtype=torch.float32, requires_grad=False))
        self.codebook_2 = torch.nn.Buffer((self.codebook ** 2).sum(dim=1)[None, :])
        self.keep_eval = False

    def estimate_eta(self, estimate_times: int = 20):
        self._compiled = False
        eta_diffs = []
        for _ in range(estimate_times):
            latents = torch.randn(1024, self.codebook_dim, dtype=self.codebook.dtype, device=self.codebook.device)
            self.eval()
            q_latents, *_ = self(latents, update_probability=0.0, )
            self.train()
            p_latents, *_ = self(latents, update_probability=0.0, )
            perturbation_scale = (p_latents - latents).square().sum(dim=-1).sqrt()
            quantization_scale = (q_latents - latents).square().sum(dim=-1).sqrt()
            mask = (perturbation_scale > 1e-6)
            eta_diff = quantization_scale[mask].mean() / perturbation_scale[mask].mean()
            eta_diffs.append(eta_diff.item())
        return sum(eta_diffs) / len(eta_diffs)

    def __repr__(self):
        return (f"VectorPerturbation("
                f"codebook_dim={self.codebook_dim}, "
                f"codebook_size={self.codebook_size}, "
                f"eta={self.eta}, "
                f"cluster_size_rate={self.cluster_size_rate}, "
                f")")

    def update_samples_queue(self, z: torch.Tensor, update_probability: float = 0.1):
        if update_probability <= 0.0:
            return
        update_mask = torch.rand(z.shape[0], device=z.device) < update_probability
        self.samples_queue.add(z[update_mask].detach())
        self._compiled = False

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if (not mode) and (not self._compiled):
            self.codebook.copy_(self.codebook_compiler.fit(self.samples_queue.samples))
            self.codebook_2.copy_((self.codebook ** 2).sum(dim=1)[None, :])
            self._compiled = True
        return self

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z2 = (z ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        dist2 = z2 + self.codebook_2 - 2 * z @ self.codebook.t()  # [N, K]
        min_sse, min_ind = dist2.min(dim=1)
        return self.codebook[min_ind], min_ind, min_sse

    def get_perturbations(self, z: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            perturbations: torch.Tensor
            perturbation_mask: torch.Tensor
            perturbation_scale: torch.Tensor
        """
        d_z = self.samples_queue.min_k_norm(z, k=self.sample_num_per_cluster, chunk_size=self.chunk_size)
        p_max_norm = self.eta * d_z[:, -1]
        perturbations, p_rad = v_utils.sample_unit_ball_uniform(
            dim=self.codebook_dim, num_samples=z.shape[0],
            device=z.device, dtype=z.dtype, eps=eps, return_radius=True)
        perturbations, p_rad = p_max_norm[:, None] * perturbations, p_max_norm * p_rad
        p_z = z + perturbations
        d_pz = self.samples_queue.min_k_norm(p_z, k=self.sample_num_per_cluster, chunk_size=self.chunk_size)
        dk_pz = d_pz[:, self.k4nn - 1]
        dk_z = d_z[:, self.k4nn - 1]
        log_accept_ratio = self.codebook_dim * (dk_z.log() + d_z[:, -1].log() - dk_pz.log() - d_pz[:, -1].log())
        accept_mask = (torch.rand_like(log_accept_ratio).clamp_min(eps).log() < log_accept_ratio)
        accept_mask &= (p_rad < self.eta * d_pz[:, -1])
        return perturbations, accept_mask, p_rad

    def forward(self, features: torch.Tensor,
                update_probability: float = 1e-2,
                eps: float = 1e-6, ) \
            -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            features: torch.Tensor, shape (batch_size, ...num_tokens..., codebook_dim)
            update_probability: the rate of updating the codebook
            eps: the epsilon value

        Returns:
            q_features: torch.Tensor, shape (batch_size, num_tokens, codebook_dim)
            indices: if return_indices: torch.Tensor, shape (batch_size, num_tokens)
                     else: torch.zeros(batch_size, num_tokens)
            others: dict
        """
        feature_shape = features.shape
        z = features.reshape(-1, feature_shape[-1])
        z, norm_loss, norm_info = self.vector_norm(z)

        with torch.no_grad():
            if self.keep_eval or (not self.training):
                q_z, indices, quantize_sse = self.quantize(z)
                z_bias = (q_z - z).detach()
                other_info = {
                    'quantize_mean_sse': quantize_sse.mean(),
                }
            else:
                perturbations, accept_mask, p_scale = self.get_perturbations(z, eps=eps)
                z_bias = (perturbations * accept_mask[:, None].to(dtype=perturbations.dtype)).detach()
                indices = torch.zeros(z.shape[0], device=z.device, dtype=torch.int64)
                other_info = {"p_accept_prob": accept_mask.float().mean()}

                self.update_samples_queue(z, update_probability=update_probability)

        q_z = z + z_bias
        q_features = q_z.view(*feature_shape)
        indices = indices.view(*feature_shape[:-1])
        return q_features, indices, {
            'norm_loss': norm_loss,
            'norm_info': norm_info,
            **other_info,
        }

    def indices_to_codes(self, indices: torch.Tensor):
        return self.codebook[indices].detach().clone()
