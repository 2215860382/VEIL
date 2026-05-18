"""Pre-allocate remaining GPU memory after models are loaded.

Usage:
    from src.utils.gpu_lock import lock_gpu, release_gpu

    # After loading all models on a device:
    _lock = lock_gpu("cuda:0", reserve_fraction=0.95)

    # ... do your work ...

    # Optional: release before exit so others can use the GPU
    release_gpu(_lock)
"""
from __future__ import annotations

import torch

_log = None


def _get_log():
    global _log
    if _log is None:
        from src.utils.logging import get_logger
        _log = get_logger("gpu_lock")
    return _log


def lock_gpu(device: str, reserve_fraction: float = 0.95) -> torch.Tensor | None:
    """Allocate reserve_fraction of the currently free GPU memory.

    Call this after all models are loaded. The returned tensor must be kept
    alive (stored in a variable) for the lock to hold.
    Returns None if device is CPU or allocation fails.
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        free, total = torch.cuda.mem_get_info(torch.device(device))
        reserve = int(free * reserve_fraction)
        if reserve < 1024 * 1024:  # less than 1 MB, don't bother
            return None
        # float32 = 4 bytes per element
        dummy = torch.empty(reserve // 4, dtype=torch.float32, device=device)
        locked_gb = reserve / 1024 ** 3
        free_gb   = free   / 1024 ** 3
        total_gb  = total  / 1024 ** 3
        _get_log().info(
            "gpu_lock: locked %.1f GB on %s (was %.1f GB free / %.1f GB total)",
            locked_gb, device, free_gb, total_gb,
        )
        return dummy
    except Exception as e:
        _get_log().warning("gpu_lock: failed to lock %s: %s", device, e)
        return None


def release_gpu(lock: torch.Tensor | None) -> None:
    """Release a previously acquired GPU lock."""
    if lock is None:
        return
    device = lock.device
    del lock
    torch.cuda.empty_cache()
    _get_log().info("gpu_lock: released lock on %s", device)
