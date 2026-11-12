"""用相同的小网络比较预测误差训练和组合效用训练。"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn


class MeanPredictor(nn.Module):
    def __init__(self, n_features, n_assets):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32), nn.Tanh(),
            nn.Linear(32, n_assets), nn.Tanh(),
        )

    def forward(self, features):
        # 限制年化均值预测的尺度，减轻噪声预测引起的极端配置。
        return 0.25 * self.network(features)


@dataclass
class FittedModel:
    model: MeanPredictor
    center: np.ndarray
    scale: np.ndarray
    metadata: dict

    def predict_mean(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.center) or not np.isfinite(values).all():
            raise ValueError("Features must be finite and match the training dimension.")
        normalized = torch.as_tensor((values - self.center) / self.scale)
        self.model.eval()
        with torch.no_grad():
            return self.model(normalized).numpy()


def buy_and_hold_returns(weights, asset_returns):
    """初始权重确定后，持有期间随资产涨跌自然漂移。"""
    asset_wealth = torch.cumprod(1 + asset_returns, dim=1)
    wealth = torch.einsum("bhn,bn->bh", asset_wealth, weights)
    previous = torch.cat((torch.ones_like(wealth[:, :1]), wealth[:, :-1]), dim=1)
    return wealth / previous - 1


def _loss(model, batch, mode, layer, risk_aversion):
    features, factors, future = batch
    mean = model(features)
    if mode == "mse":
        return (mean - 252 * future.mean(dim=1)).square().mean()
    weights, = layer(mean, factors, solver_args={"eps": 1e-6, "max_iters": 10000})
    checked = weights.detach()
    if not torch.isfinite(checked).all():
        raise RuntimeError("Differentiable solver returned non-finite weights.")
    if checked.min() < -2e-4 or (checked.sum(dim=-1) - 1).abs().max() > 2e-4:
        raise RuntimeError("Differentiable solver returned infeasible weights.")
    # 清除求解器容差范围内的负权重，同时保留可微计算图。
    weights = weights.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    returns = buy_and_hold_returns(weights, future)
    utility = 252 * (returns.mean(dim=1) - risk_aversion * returns.var(dim=1, unbiased=False))
    return -utility.mean()


def _check_dataset(dataset, name):
    features, factors, future = (np.asarray(value, dtype=float) for value in dataset)
    if features.ndim != 2 or factors.ndim != 3 or future.ndim != 3:
        raise ValueError(f"Invalid {name} dimensions.")
    count, n_assets = len(features), future.shape[-1]
    if count == 0 or len(future) != count or factors.shape != (count, n_assets, n_assets):
        raise ValueError(f"Inconsistent or empty {name} samples.")
    if future.shape[1] < 2 or n_assets < 2:
        raise ValueError("Training requires at least two assets and two holding days.")
    if not all(np.isfinite(value).all() for value in (features, factors, future)):
        raise ValueError(f"Non-finite values in {name}.")
    if np.any(future <= -1):
        raise ValueError("Daily simple returns must exceed -1.")
    return features, factors, future


def _tensor_dataset(dataset, center, scale):
    features, factors, future = dataset
    return tuple(torch.as_tensor(value, dtype=torch.float64) for value in (
        (features - center) / scale, factors, future,
    ))


def _validation_loss(model, dataset, mode, layer, risk_aversion, batch_size):
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset[0]), batch_size):
            batch = tuple(value[start:start + batch_size] for value in dataset)
            loss = _loss(model, batch, mode, layer, risk_aversion)
            total += float(loss) * len(batch[0])
    return total / len(dataset[0])


def _fit_epochs(model, train, validation, mode, layer, risk_aversion, epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best_state, best_loss, best_epoch = None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(train[0])).split(batch_size):
            batch = tuple(value[indices] for value in train)
            optimizer.zero_grad()
            loss = _loss(model, batch, mode, layer, risk_aversion)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        score = _validation_loss(model, validation, mode, layer, risk_aversion, batch_size)
        if not np.isfinite(score):
            raise RuntimeError("Validation produced a non-finite loss.")
        if score < best_loss:
            best_state, best_loss, best_epoch = deepcopy(model.state_dict()), score, epoch + 1
    model.load_state_dict(best_state)
    return best_loss, best_epoch


def train_model(train, validation, mode, risk_aversion=5.0, epochs=15, seed=7, batch_size=32):
    """验证集只用于选取训练轮次，标准化参数完全来自训练集。"""
    if mode not in {"mse", "utility"} or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training mode or hyperparameters.")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be finite and positive.")
    train = _check_dataset(train, "training")
    validation = _check_dataset(validation, "validation")
    if any(a.shape[1:] != b.shape[1:] for a, b in zip(train, validation)):
        raise ValueError("Training and validation dimensions must match.")
    started = perf_counter()
    torch.manual_seed(seed)
    center = train[0].mean(axis=0)
    scale = train[0].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    train_tensors = _tensor_dataset(train, center, scale)
    validation_tensors = _tensor_dataset(validation, center, scale)
    model = MeanPredictor(train[0].shape[-1], train[2].shape[-1]).double()
    layer = None
    if mode == "utility":
        from .optimization import make_allocation_layer
        layer = make_allocation_layer(train[2].shape[-1], risk_aversion)
    best_loss, best_epoch = _fit_epochs(
        model, train_tensors, validation_tensors, mode, layer, risk_aversion, epochs, batch_size,
    )
    metadata = {
        "mode": mode, "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
        "validation_loss": best_loss, "train_seconds": perf_counter() - started,
        "training_samples": len(train[0]), "validation_samples": len(validation[0]),
    }
    return FittedModel(model.eval(), center, scale, metadata)
