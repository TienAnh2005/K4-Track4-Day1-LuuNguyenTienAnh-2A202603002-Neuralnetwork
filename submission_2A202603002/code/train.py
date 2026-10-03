"""train.py — Vòng lặp huấn luyện, đánh giá và thực nghiệm.

Chức năng:
  - set_seed cho tính tái lập
  - evaluate, predict, compute_loss
  - run_experiment(cfg, data) cho mọi thí nghiệm
  - write_predictions và final_eval phục vụ chấm điểm eval
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

N_CLASSES = 7

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # chọn bằng val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch, torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp từ ma trận nhầm lẫn 7x7."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Tính loss: 'ce' (CrossEntropy) hoặc 'mse' (MSE với one-hot)."""
    loss_clean = loss_name.lower().strip()
    if loss_clean == "ce":
        return F.cross_entropy(logits, y)
    elif loss_clean == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát {loss_name} không hợp lệ. Chọn 'ce' hoặc 'mse'.")


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    was_training = model.training
    model.eval()

    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))

    if was_training:
        model.train()

    return torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.int64, device=X.device)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, loss_name: str = "ce",
             batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() và no_grad."""
    was_training = model.training
    model.eval()

    n = len(X)
    if n == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}

    total_loss = 0.0
    all_preds = []
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * len(xb)
        all_preds.append(torch.argmax(logits, dim=1))

    preds = torch.cat(all_preds, dim=0).cpu().numpy()
    targets = y.cpu().numpy()

    acc = float((preds == targets).mean())

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    if was_training:
        model.train()

    return {
        "loss": float(total_loss / n),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg.get("seed", 42))

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_method = cfg.get("init", "he")
    loss_name = cfg.get("loss", "ce")
    opt_name = cfg.get("optimizer", "sgd_momentum")
    lr = float(cfg.get("lr", 0.05))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    batch_size = int(cfg.get("batch", 512))
    epochs = int(cfg.get("epochs", 20))
    clip_norm = cfg.get("clip_norm", None)
    precision = cfg.get("precision", "fp32").lower().strip()

    device = data["X_tr"].device
    model = MLP(hidden=hidden, dropout=dropout, init=init_method).to(device)

    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số không khớp: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(opt_name, model, lr=lr, weight_decay=weight_decay, momentum=momentum)

    use_cuda = (device.type == "cuda")
    scaler = None
    if precision == "fp16" and use_cuda:
        scaler = torch.amp.GradScaler("cuda")

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Đo loss bước 0 trên val (trước cập nhật đầu tiên)
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = step0_res["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    diverged = False

    # Để tính train_loss nhanh ở eval mode, cố định tập con 50.000 mẫu của train nếu train quá lớn
    eval_tr_n = min(50000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:eval_tr_n]
    y_tr_eval = data["y_tr"][:eval_tr_n]

    for ep in range(1, epochs + 1):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        batch_norms = []
        gen = torch.Generator(device=device).manual_seed(cfg.get("seed", 42) + ep)

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            if precision in ("fp16", "bf16") and use_cuda:
                amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model, clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model, clip_norm)
                optimizer.step()

            batch_norms.append(gn)

        if use_cuda:
            torch.cuda.synchronize()
        ep_time = time.time() - t0

        if diverged:
            print(f"[{cfg.get('exp_id')}] Epoch {ep}: Loss bị NaN/inf -> dừng sớm!")
            break

        # Đánh giá cuối epoch ở chế độ eval()
        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)

        mean_gn = float(np.mean(batch_norms)) if batch_norms else 0.0

        history["epoch"].append(ep)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(ep_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())

    # Tổng hợp metric tại best epoch
    if history["epoch"]:
        best_idx = best_epoch - 1
        summary_val_acc = history["val_acc"][best_idx]
        summary_val_macro_f1 = history["val_macro_f1"][best_idx]
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        time_per_epoch = float(np.mean(history["epoch_time_s"]))
    else:
        summary_val_acc = 0.0
        summary_val_macro_f1 = 0.0
        final_train_loss = float("inf")
        final_val_loss = float("inf")
        time_per_epoch = 0.0

    peak_mem_MB = 0.0
    if use_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(summary_val_acc),
        "val_macro_f1": float(summary_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề row_id,pred."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "row_id": row_id.astype(int),
        "pred": preds.astype(int),
    })
    df.to_csv(path, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán eval vào {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_method = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init_method).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds
