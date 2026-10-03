"""optimizer.py — Bộ tối ưu hoá và cắt gradient.

Hỗ trợ: SGD, SGD+Momentum, Adam, AdamW và Gradient Clipping.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    name_clean = name.lower().strip()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Optimizer {name} không hợp lệ. Chọn từ {OPTIMIZERS}")

    # Lọc params nếu truyền model thay vì parameters
    if hasattr(params, "parameters"):
        params = params.parameters()
    param_list = [p for p in params if p.requires_grad]

    if name_clean == "sgd":
        return torch.optim.SGD(param_list, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return torch.optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return torch.optim.Adam(param_list, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return torch.optim.AdamW(param_list, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học."""
    if name is None:
        return None
    name_clean = name.lower().strip()
    if name_clean == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 1e-6))
    return None


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt."""
    if hasattr(params, "parameters"):
        params = params.parameters()
    param_list = [p for p in params if p.grad is not None]
    if not param_list:
        return 0.0

    if max_norm is None or max_norm <= 0:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float(max_norm))

    return float(total_norm)
