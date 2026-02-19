import logging
import threading

import torch

from . import ddp

log = logging.getLogger("VectorPerturbation.v_utils")


class SamplesQueue(torch.nn.Module):
    def __init__(self, init_samples: torch.Tensor):
        super().__init__()
        self.len, _ = init_samples.shape
        self.samples = init_samples.clone().detach()
        self.right_index: int = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self.len

    def _apply(self, fn, recurse=True):
        super()._apply(fn)
        self.samples = fn(self.samples)
        return self

    def _build_slices(self, start: int, length: int):
        if length > self.len:
            raise ValueError(f'length({length}) must be less than max_len({self.len})')

        end = start + length
        if end <= self.len:
            return [slice(start, end), ]
        else:
            return [slice(start, self.len), slice(0, end - self.len)]

    def _set(self, start: int, data: torch.Tensor):
        set_len = len(data)
        if set_len > self.len:
            data = data[-self.len:]
            set_len = self.len
        for s in self._build_slices(start, set_len):
            s_len = s.stop - s.start
            self.samples[s] = data[:s_len]
            data = data[s_len:]
        return set_len

    def min_k_norm(self, query_data: torch.Tensor, k: int, chunk_size: int = 1024) -> torch.Tensor:

        def compute_chunk_topk(q_chunk, s_all, k_val):
            dists = torch.cdist(q_chunk, s_all, p=2)
            topk_dists, _ = torch.topk(dists, k=k_val, dim=1, largest=False)
            return topk_dists

        compiled_fn = torch.compile(compute_chunk_topk, mode='max-autotune', dynamic=True)

        topk_results = torch.empty(query_data.shape[0], k, dtype=query_data.dtype, device=query_data.device)
        for i in range(0, query_data.shape[0], chunk_size):
            end_idx = min(i + chunk_size, query_data.shape[0])
            topk_results[i: end_idx] = compiled_fn(query_data[i: end_idx], self.samples, k)
        return topk_results

    def add(self, data: torch.Tensor):
        with self._lock:
            set_len = self._set(self.right_index, data)
            self.right_index = (self.right_index + set_len) % self.len


class SamplesKMeans:
    def __init__(self, n_clusters: int, max_iter: int = 20, tol: float = 1e-4,
                 batch_size: int = 16384):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.centroids = None

    def _initialize_centroids(self, data: torch.Tensor):
        if ddp.is_process_zero():
            # random init
            # idx = torch.randperm(data.size(0), device=data.device)[:self.n_clusters]
            # self.centroids = data[idx].clone().contiguous()
            # kmeans++
            self.centroids = torch.empty((self.n_clusters, data.size(1)), dtype=data.dtype, device=data.device)
            first_idx = torch.randint(0, data.size(0), (1,)).item()
            self.centroids[0] = data[first_idx]
            current_center = self.centroids[0].unsqueeze(0)  # (1, d)
            current_min_dist_sq = torch.sum((data - current_center) ** 2, dim=1)
            for i in range(1, self.n_clusters):
                limit = torch.quantile(current_min_dist_sq, 0.90)
                weights = torch.clamp(current_min_dist_sq, max=limit)
                if weights.sum() == 0:
                    next_idx = torch.randint(0, data.size(0), (1,)).item()
                else:
                    next_idx = torch.multinomial(weights, 1)

                new_center = data[next_idx]  # (1, d)
                self.centroids[i] = new_center

                new_dist_sq = torch.sum((data - new_center) ** 2, dim=1)
                current_min_dist_sq = torch.minimum(current_min_dist_sq, new_dist_sq)

        else:
            self.centroids = torch.empty((self.n_clusters, data.size(1)), dtype=data.dtype, device=data.device)

        ddp.broadcast(self.centroids, src=0)

    def _assign_chunked(self, data: torch.Tensor) -> torch.Tensor:
        n_samples = data.size(0)
        assignments = torch.empty(n_samples, dtype=torch.int64, device=data.device)
        centroids_t = self.centroids.t().contiguous()
        centroids_norm = (centroids_t ** 2).sum(dim=0, keepdim=True)

        def compute_batch_assignment(q, c_t, c_norm):
            q_norm = (q ** 2).sum(dim=1, keepdim=True)
            dot_product = torch.mm(q, c_t)
            distances = q_norm + c_norm - 2 * dot_product
            return torch.argmin(distances, dim=1)

        compiled_kernel = torch.compile(compute_batch_assignment, mode='max-autotune', dynamic=True)

        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            batch_data = data[i:end]
            assignments[i:end] = compiled_kernel(batch_data, centroids_t, centroids_norm)

        return assignments

    def _update_centroids(self, cluster_sums: torch.Tensor, cluster_counts: torch.Tensor):
        nonempty_mask = cluster_counts > 0
        self.centroids[nonempty_mask] = cluster_sums[nonempty_mask] / cluster_counts[nonempty_mask, None]
        empty_idx = torch.nonzero(~nonempty_mask, as_tuple=True)[0]
        n_empty = empty_idx.size(0)
        if n_empty != 0 and ddp.is_process_zero():
            log.debug(f"Found {n_empty} empty clusters. Performing splitting strategy.")
            _, candidate_idx = torch.topk(cluster_counts, k=n_empty, largest=True, sorted=True)
            largest_centroids = self.centroids[candidate_idx]
            noise_scale = torch.finfo(self.centroids.dtype).resolution
            noise = torch.randn_like(largest_centroids) * noise_scale
            self.centroids[empty_idx] = largest_centroids + noise
            self.centroids[candidate_idx] = largest_centroids - noise

        ddp.broadcast(self.centroids, src=0)

    @torch.no_grad()
    def fit(self, local_data: torch.Tensor, reinit_centroids: bool = True) -> torch.Tensor:
        if ddp.is_process_zero():
            log.info(f"Starting KMeans with {self.n_clusters} clusters")
        # local_data = local_data.float()
        if (self.centroids is None) or reinit_centroids:
            self._initialize_centroids(local_data)

        for i in range(self.max_iter):
            old_centroids = self.centroids.clone()

            assignments = self._assign_chunked(local_data)

            local_counts = torch.bincount(assignments, minlength=self.n_clusters)
            ddp.all_reduce(local_counts, ddp.ReduceOp.SUM)

            local_sums = torch.zeros_like(self.centroids)
            local_sums.index_add_(0, assignments, local_data)
            ddp.all_reduce(local_sums, ddp.ReduceOp.SUM)

            self._update_centroids(local_sums, local_counts)

            # Check Convergence
            center_shift = torch.norm(self.centroids - old_centroids, p='fro')
            if ddp.is_process_zero():
                log.debug(f"Iter {i}: Centroid shift = {center_shift.item():.6f}")
            if center_shift.item() < self.tol:
                break

        if ddp.is_process_zero():
            log.info(f"Finished KMeans")
        return self.centroids


def sample_unit_ball_uniform(dim, num_samples, device='cpu', dtype=torch.float32, eps=1e-6, return_radius=False):
    """
    Generates points uniformly distributed inside the unit d-dimensional ball.
    """
    normal_samples = torch.randn(num_samples, dim, device=device, dtype=dtype)
    norms = torch.norm(normal_samples, p=2, dim=-1, keepdim=True)
    uniform_on_surface = normal_samples / norms.clamp_min(eps)
    random_rad = (torch.rand(num_samples, device=device, dtype=dtype) ** (1.0 / dim))
    uniform_in_ball = uniform_on_surface * random_rad[:, None]
    if return_radius:
        return uniform_in_ball, random_rad
    else:
        return uniform_in_ball


def samples_queue_test():
    q = SamplesQueue(torch.zeros(5, 2))
    print(q.samples)
    for x in [
        torch.ones(3, 2),
        torch.ones(3, 2) * 2,
        torch.ones(4, 2) * 3,
        torch.ones(5, 2) * 4,
        torch.ones(6, 2) * 5,
    ]:
        q.add(x)
        print(f"add len: {len(x)}")
        print(q.samples)
        print(q.min_k_norm(torch.zeros(2, 2), k=4))


if __name__ == '__main__':
    samples_queue_test()
