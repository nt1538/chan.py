"""XGBoost baseline and PyTorch sequence-LSTM implementations."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier, XGBRegressor


def make_xgboost(config) -> XGBRegressor | XGBClassifier:
    """Create the reproducible tabular baseline."""
    model_class = XGBClassifier if config.target_mode == "up_direction" else XGBRegressor
    objective = "binary:logistic" if config.target_mode == "up_direction" else "reg:squarederror"
    return model_class(
        objective=objective, tree_method="hist",
        n_estimators=config.n_estimators, max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate, subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        min_child_weight=config.xgb_min_child_weight,
        reg_alpha=config.xgb_reg_alpha, reg_lambda=config.xgb_reg_lambda,
        n_jobs=config.n_jobs, random_state=config.random_seed,
    )


class ExtremeLSTM(nn.Module):
    """Many-to-one LSTM supporting extreme or unrestricted return targets."""

    def __init__(self, n_features: int, hidden_size: int, layers: int, dropout: float,
                 loss_as_magnitude: bool = False, n_targets: int = 2,
                 constrain_extremes: bool = True):
        super().__init__()
        self.loss_as_magnitude = bool(loss_as_magnitude)
        self.constrain_extremes = bool(constrain_extremes)
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                  nn.Linear(hidden_size // 2, int(n_targets)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(values)
        raw = self.head(sequence[:, -1])
        # Encode the financial constraints directly in the network: maximum
        # gain cannot be negative and signed maximum loss cannot be positive.
        if not self.constrain_extremes:
            return raw
        loss = F.softplus(raw[:, 1])
        return torch.stack((F.softplus(raw[:, 0]), loss if self.loss_as_magnitude else -loss), dim=1)


class SequenceDataset(Dataset):
    """Lazy rolling windows avoid materializing a huge three-dimensional array."""

    def __init__(self, x: np.ndarray, y: np.ndarray, length: int, stride: int = 1):
        self.x, self.y, self.length = x, y, int(length)
        self.ends = np.arange(self.length - 1, len(x), max(1, int(stride)))

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        return torch.from_numpy(self.x[end - self.length + 1:end + 1]), torch.from_numpy(self.y[end])


def fit_lstm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, config) -> Dict:
    """Fit with validation-model selection and return a portable state bundle."""
    torch.manual_seed(config.random_seed)
    median = np.nanmedian(x_train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(x_train), x_train, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def transform(x):
        x = np.where(np.isfinite(x), x, median)
        return np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    train_ds = SequenceDataset(transform(x_train), y_train.astype(np.float32), config.lstm_sequence_length, config.lstm_train_stride)
    val_ds = SequenceDataset(transform(x_val), y_val.astype(np.float32), config.lstm_sequence_length)
    if not len(train_ds) or not len(val_ds):
        raise ValueError("LSTM split is shorter than lstm_sequence_length")
    constrain_extremes = str(getattr(config, "target_mode", "extremes")) == "extremes"
    model = ExtremeLSTM(x_train.shape[1], config.lstm_hidden_size, config.lstm_layers,
                        config.lstm_dropout, loss_as_magnitude=constrain_extremes,
                        n_targets=y_train.shape[1], constrain_extremes=constrain_extremes)
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=config.lstm_learning_rate), nn.SmoothL1Loss()
    best_loss, best_state = float("inf"), None
    safe_batch = max(1, int(config.lstm_max_batch_feature_values) // max(1, config.lstm_sequence_length * x_train.shape[1]))
    effective_batch = min(int(config.lstm_batch_size), safe_batch)
    if effective_batch < int(config.lstm_batch_size):
        print(f"[LSTM] Reducing batch_size {config.lstm_batch_size} -> {effective_batch} "
              f"for sequence_length={config.lstm_sequence_length}, features={x_train.shape[1]} to protect memory.")
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=effective_batch, shuffle=False)
    for epoch in range(config.lstm_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval(); losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                losses.append(float(loss_fn(model(xb), yb)))
        current = float(np.mean(losses))
        if current < best_loss:
            best_loss = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if getattr(config, "verbose", False):
            print(
                f"[LSTM] Epoch {epoch + 1}/{config.lstm_epochs} | "
                f"train loss={float(np.mean(train_losses)):.6f} | "
                f"validation loss={current:.6f} | best={best_loss:.6f}",
                flush=True,
            )
    return {"state_dict": best_state, "median": median, "mean": mean, "scale": scale,
            "n_features": x_train.shape[1], "sequence_length": config.lstm_sequence_length,
            "hidden_size": config.lstm_hidden_size, "layers": config.lstm_layers,
            "dropout": config.lstm_dropout, "validation_loss": best_loss,
            "effective_batch_size": effective_batch,
            "loss_as_magnitude": constrain_extremes,
            "n_targets": int(y_train.shape[1]),
            "constrain_extremes": constrain_extremes}


def predict_lstm(bundle: Dict, x: np.ndarray, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Return valid ending-row indices and corresponding two-target predictions."""
    model = ExtremeLSTM(bundle["n_features"], bundle["hidden_size"], bundle["layers"],
                        bundle["dropout"], bundle.get("loss_as_magnitude", False),
                        bundle.get("n_targets", 2), bundle.get("constrain_extremes", True))
    model.load_state_dict(bundle["state_dict"]); model.eval()
    values = np.where(np.isfinite(x), x, bundle["median"])
    values = np.nan_to_num((values - bundle["mean"]) / bundle["scale"]).astype(np.float32)
    n_targets = int(bundle.get("n_targets", 2))
    dataset = SequenceDataset(values, np.zeros((len(values), n_targets), dtype=np.float32), bundle["sequence_length"])
    outputs = []
    effective_batch = min(int(batch_size), int(bundle.get("effective_batch_size", batch_size)))
    with torch.no_grad():
        for xb, _ in DataLoader(dataset, batch_size=effective_batch, shuffle=False):
            outputs.append(model(xb).numpy())
    pred = np.concatenate(outputs) if outputs else np.empty((0, n_targets))
    return dataset.ends, pred
