"""data.py — Chuẩn bị và xử lý dữ liệu CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    p = Path(processed_dir)
    train_path = p / "train.npz"
    eval_path = p / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file npz trong {processed_dir}. Hãy chạy 'python scripts/split_data.py' trước."
        )

    tr_data = np.load(train_path)
    ev_data = np.load(eval_path)

    X_train_full = tr_data["X"].astype(np.float32)
    y_train_full = tr_data["y"].astype(np.int64)
    X_eval = ev_data["X"].astype(np.float32)
    y_eval = ev_data["y"].astype(np.int64)
    eval_row_id = ev_data["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == N_FEATURES
    assert X_eval.ndim == 2 and X_eval.shape[1] == N_FEATURES
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full)
    assert y_eval.ndim == 1 and len(y_eval) == len(X_eval)
    assert 0 <= y_train_full.min() and y_train_full.max() < N_CLASSES
    assert 0 <= y_eval.min() and y_eval.max() < N_CLASSES

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed, shuffle=True
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val)."""
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    t_y_tr = torch.tensor(y_tr, dtype=torch.int64, device=dev)
    t_X_val = torch.tensor(X_val, dtype=torch.float32, device=dev)
    t_y_val = torch.tensor(y_val, dtype=torch.int64, device=dev)
    t_X_eval = torch.tensor(X_eval, dtype=torch.float32, device=dev)
    t_y_eval = torch.tensor(y_eval, dtype=torch.int64, device=dev)

    # Thống kê cơ bản
    ctr_val = np.bincount(y_val, minlength=N_CLASSES)
    majority_acc = ctr_val.max() / len(y_val)
    print(f"Data prepared on {dev}:")
    print(f"  Train: X={t_X_tr.shape}, y={t_y_tr.shape}")
    print(f"  Val  : X={t_X_val.shape}, y={t_y_val.shape}")
    print(f"  Eval : X={t_X_eval.shape}, y={t_y_eval.shape}")
    print(f"  Majority class accuracy on val: {majority_acc:.4f} (class {ctr_val.argmax()})")

    return {
        "X_tr": t_X_tr,
        "y_tr": t_y_tr,
        "X_val": t_X_val,
        "y_val": t_y_val,
        "X_eval": t_X_eval,
        "y_eval": t_y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
