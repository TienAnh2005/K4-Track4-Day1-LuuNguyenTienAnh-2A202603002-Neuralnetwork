"""plots.py — Vẽ biểu đồ huấn luyện và so sánh thí nghiệm.

Mỗi thí nghiệm có một ảnh figures/<exp_id>.png (3 ô).
Mỗi nhóm thí nghiệm có một ảnh figures/compare_<nhóm>.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result.get("cfg", {})
    hist = result.get("history", {})
    summary = result.get("summary", {})

    exp_id = cfg.get("exp_id", "experiment")
    opt_name = cfg.get("optimizer", "")
    lr = cfg.get("lr", "")
    batch = cfg.get("batch", "")
    best_epoch = summary.get("best_epoch", 1)

    epochs = hist.get("epoch", list(range(1, len(hist.get("val_loss", [])) + 1)))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Train & Val Loss
    ax1 = axes[0]
    if "train_loss" in hist and len(hist["train_loss"]) > 0:
        ax1.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    if "val_loss" in hist and len(hist["val_loss"]) > 0:
        ax1.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Ô 2: Val Accuracy & Macro-F1
    ax2 = axes[1]
    if "val_acc" in hist and len(hist["val_acc"]) > 0:
        ax2.plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", lw=2)
    if "val_macro_f1" in hist and len(hist["val_macro_f1"]) > 0:
        ax2.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", lw=2, linestyle="-.")
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Ô 3: Gradient Norm (trước khi clip)
    ax3 = axes[2]
    if "grad_norm" in hist and len(hist["grad_norm"]) > 0:
        ax3.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#9467bd", lw=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.set_title("Gradient Norm Dynamics")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle(f"{exp_id} | Opt: {opt_name}, LR: {lr}, Batch: {batch}, Init: {cfg.get('init')}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    if not results:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metric_labels = {
        "val_loss": "Validation Loss",
        "val_acc": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "train_loss": "Train Loss",
        "grad_norm": "Gradient Norm (pre-clip)",
    }
    label_y = metric_labels.get(metric, metric)

    for r in results:
        cfg = r.get("cfg", {})
        hist = r.get("history", {})
        exp_id = cfg.get("exp_id", "exp")
        vals = hist.get(metric, [])
        epochs = hist.get("epoch", list(range(1, len(vals) + 1)))
        if vals:
            ax.plot(epochs, vals, label=exp_id, lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(label_y)
    ax.set_title(title or f"Comparison: {label_y}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
