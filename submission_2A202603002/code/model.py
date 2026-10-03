"""model.py — Định nghĩa kiến trúc MLP và khởi tạo tham số.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):
    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)
"""
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định chuẩn."""

    def __init__(self, hidden: tuple[int, ...] = (256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout_rate = dropout
        self.init_name = init

        layers = []
        curr_in = in_features
        for h in hidden:
            layers.append(nn.Linear(curr_in, h, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            curr_in = h

        layers.append(nn.Linear(curr_in, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32 -> logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias luôn = 0)."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif init == "default":
                pass
            else:
                raise ValueError(f"Khởi tạo không hợp lệ: {init}. Chọn zeros, normal, xavier, he, hoặc default.")


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt sau mỗi lớp Linear ở bước 0 (một lô val)."""
    was_training = model.training
    model.eval()

    h = x
    stds = []
    # Đi qua từng layer trong net
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        for layer in model.net:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                stds.append(float(h.std().item()))
    else:
        for m in model.children():
            h = m(h)
            stds.append(float(h.std().item()))

    if was_training:
        model.train()

    return stds
