"""Автовыбор устройства PyTorch и настроек MuJoCo под хост (macOS / Linux, CUDA / CPU / MPS)."""

from __future__ import annotations

import platform
from typing import Any, Optional


def apply_mjx_overrides_to_playground_cfg(env_cfg: Any) -> Any:
    """На CPU переключает MJX бэкенд Playground с warp (требует CUDA) на jax.

    Вызывать после ``get_default_config`` и до ``registry.load``.
    На GPU-хосте ничего не делает.
    """
    import jax

    try:
        if jax.devices("cuda"):
            return env_cfg
    except RuntimeError:
        pass

    with env_cfg.unlocked():
        env_cfg.impl = "jax"

    return env_cfg


def default_torch_device(explicit: Optional[str] = None) -> str:
    """Возвращает явное устройство или лучший доступный: cuda → mps → cpu."""
    if explicit is not None:
        return explicit
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if platform.system() == "Darwin":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"


def rslrl_brax_wrapper_jax_device_rank(torch_device: str) -> Optional[int]:
    """device_rank для RSLRLBraxWrapper: GPU rank при CUDA, None иначе."""
    if "cuda" in torch_device:
        return int(torch_device.split(":")[-1])
    return None


def default_num_envs(explicit: Optional[int] = None) -> int:
    """Параллельные среды: 4096 на GPU, 64 на CPU/MPS."""
    if explicit is not None:
        return explicit
    import torch

    if torch.cuda.is_available():
        return 4096
    return 64
