from functools import lru_cache
from typing import Callable

import torch
from torch import distributed
from torch.distributed import ReduceOp


@lru_cache
def get_world_size():
    return distributed.get_world_size()


@lru_cache
def is_distributed():
    return distributed.is_initialized() and get_world_size() > 1


@lru_cache
def is_process_zero():
    return (not is_distributed()) or (distributed.get_rank() == 0)


def broadcast(local_tensor: torch.Tensor, **kwargs):
    if is_distributed():
        distributed.broadcast(local_tensor, **kwargs)


def all_reduce(local_tensor: torch.Tensor, reduce_op, **kwargs) -> torch.Tensor:
    if is_distributed():
        torch.distributed.all_reduce(local_tensor, reduce_op, **kwargs)


def all_gather(local_tensor: torch.Tensor, tensor_size: list[int] = None) -> list[torch.Tensor]:
    if not is_distributed():
        return [local_tensor, ]
    if tensor_size is None:
        local_size = torch.tensor(local_tensor.shape, dtype=torch.int64, device=local_tensor.device)
        all_sizes = [torch.empty_like(local_size) for _ in range(get_world_size())]
        distributed.all_gather(all_sizes, local_size)
        all_sizes = [t_size.tolist() for t_size in all_sizes]
    else:
        all_sizes = [tensor_size for _ in range(get_world_size())]
    all_tensors = [torch.empty(t_size, dtype=local_tensor.dtype, device=local_tensor.device)
                   for t_size in all_sizes]
    distributed.all_gather(all_tensors, local_tensor)
    return all_tensors


def tensor_gather(func):
    def wrapper(*args, **kwargs) -> list:
        result_tensor = func(*args, **kwargs)
        gathered_tensors = all_gather(result_tensor)
        return gathered_tensors

    return wrapper


def tensor_reducer(reduction='sum', **reduce_args):
    reduce_op = getattr(ReduceOp, reduction.upper())

    def decorator(func):
        def wrapper(*args, **kwargs) -> torch.Tensor:
            result_tensor = func(*args, **kwargs).clone()
            all_reduce(result_tensor, reduce_op, **reduce_args)
            return result_tensor

        return wrapper

    return decorator


def gather_objects_on_main_process(local_obj):
    if not is_distributed():
        return [local_obj, ]
    gathered_objs = [None for _ in range(get_world_size())]
    distributed.gather_object(
        local_obj,
        gathered_objs if is_process_zero() else None,
        dst=0
    )
    return gathered_objs


def gather_objects_on_all_processes(local_obj):
    if not is_distributed():
        return [local_obj, ]
    gathered_objs = [None for _ in range(get_world_size())]
    distributed.all_gather_object(gathered_objs, local_obj)
    return gathered_objs


def object_gather_on_main_process(func: Callable[..., None]):
    def wrapper(*args, **kwargs) -> list:
        result_obj = func(*args, **kwargs)
        gathered_objs = gather_objects_on_main_process(result_obj)
        return gathered_objs

    return wrapper


def object_gather(func: Callable[..., None]):
    def wrapper(*args, **kwargs) -> list:
        result_obj = func(*args, **kwargs)
        gathered_objs = gather_objects_on_all_processes(result_obj)
        return gathered_objs

    return wrapper


def list_gather(func):
    def wrapper(*args, **kwargs):
        gathered_objs: list[list] = object_gather(func)(*args, **kwargs)
        return [item for obj_list in gathered_objs for item in obj_list]

    return wrapper
