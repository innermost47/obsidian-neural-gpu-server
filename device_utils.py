import gc
import os
from typing import Optional

import torch


def is_mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_device() -> Optional[torch.device]:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return None


def require_device() -> torch.device:
    device = get_device()
    if device is None:
        raise RuntimeError(
            "No supported GPU available (CUDA or Apple Silicon MPS). "
            "CPU mode is not allowed."
        )
    return device


def get_dtype(device: torch.device, prefer_bf16: bool = False) -> torch.dtype:
    if device.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        mps_dtype = os.getenv("MPS_DTYPE", "float16").strip().lower()
        return torch.float32 if mps_dtype == "float32" else torch.float16
    return torch.float32


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return device.type


def free_device_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif is_mps_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def apply_mps_patches() -> None:
    if not is_mps_available():
        return

    try:
        from torchsde._brownian import brownian_interval
    except ImportError:
        return

    original_randn = brownian_interval._randn

    def mps_safe_randn(size, dtype, device, seed):
        if torch.device(device).type != "mps":
            return original_randn(size, dtype, device, seed)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(size, dtype=dtype, device="cpu", generator=generator)
        return noise.to(device)

    brownian_interval._randn = mps_safe_randn
