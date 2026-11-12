"""Learn a reusable allocator and compare its error and cost with a solver."""

import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .optimization import MeanVarianceSolver


def sample_instances(count, n_assets, rng):
    """Independent annualized instances with positive definite covariance."""
    loadings = rng.normal(size=(count, n_assets, n_assets))
    covariance = loadings @ loadings.transpose(0, 2, 1) / n_assets
    covariance += np.eye(n_assets)[None] * 0.2
    scale = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    covariance /= scale[:, :, None] * scale[:, None, :]
    volatility = rng.uniform(0.1, 0.35, (count, n_assets))
    covariance *= volatility[:, :, None] * volatility[:, None, :]
    means = rng.uniform(-0.05, 0.2, (count, n_assets))
    return means, covariance


def features(means, covariance, previous):
    return np.concatenate([means, covariance.reshape(len(means), -1), previous], axis=1)


def allocation_model(n_assets):
    return nn.Sequential(
        nn.Linear(2 * n_assets + n_assets**2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_assets),
        nn.Softmax(dim=-1),
    )


def constrain_turnover(weights, previous, limit):
    """Move along a simplex segment; its L1 length cannot exceed the limit."""
    if limit is None:
        return weights
    distance = (weights - previous).abs().sum(dim=-1, keepdim=True)
    fraction = (limit / distance.clamp_min(1e-12)).clamp(max=1)
    return previous + fraction * (weights - previous)


def fit_model(inputs, targets, n_assets, epochs, turnover_limit):
    center = inputs.mean(axis=0)
    scale = np.maximum(inputs.std(axis=0), 1e-8)
    x = torch.tensor((inputs - center) / scale, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    model = allocation_model(n_assets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        previous = torch.tensor(inputs[:, -n_assets:], dtype=torch.float32)
        weights = constrain_turnover(model(x), previous, turnover_limit)
        loss = (weights - y).square().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, center, scale


def predict(model, center, scale, means, covariance, previous, turnover_limit):
    # Include input packing and normalization in inference timing.
    inputs = (features(means, covariance, previous) - center) / scale
    with torch.no_grad():
        weights = model(torch.tensor(inputs, dtype=torch.float32))
        prior = torch.tensor(previous, dtype=torch.float32)
        return constrain_turnover(weights, prior, turnover_limit).numpy()


def benchmark(model, center, scale, means, covariance, previous, solver):
    predict(model, center, scale, means[:1], covariance[:1], previous[:1], solver.turnover_limit)
    solver.solve(means[0], covariance[0], previous[0])
    neural_times, solver_times = [], []
    for mean, cov, prior in zip(means, covariance, previous):
        start = perf_counter()
        predict(model, center, scale, mean[None], cov[None], prior[None], solver.turnover_limit)
        neural_times.append(perf_counter() - start)
        start = perf_counter()
        solver.solve(mean, cov, prior)
        solver_times.append(perf_counter() - start)
    batch_times = []
    for _ in range(20):
        start = perf_counter()
        predict(model, center, scale, means, covariance, previous, solver.turnover_limit)
        batch_times.append(perf_counter() - start)
    return {
        "nn_single_median_ms": float(np.median(neural_times) * 1000),
        "solver_single_median_ms": float(np.median(solver_times) * 1000),
        "nn_batch_instances_per_second": float(len(means) / np.median(batch_times)),
        "timing_device": "CPU",
        "torch_threads": torch.get_num_threads(),
    }


def evaluate(weights, optimum, means, covariance, risk_aversion):
    def objective(portfolios):
        risk = np.einsum("bi,bij,bj->b", portfolios, covariance, portfolios)
        return risk_aversion * risk - np.sum(means * portfolios, axis=1)

    gap = objective(weights) - objective(optimum)
    equal = np.full_like(weights, 1 / weights.shape[1])
    return {
        "objective_gap_mean": float(gap.mean()),
        "objective_gap_p95": float(np.quantile(gap, 0.95)),
        "objective_gap_max": float(gap.max()),
        "equal_weight_objective_gap_mean": float((objective(equal) - objective(optimum)).mean()),
        "weight_rmse": float(np.sqrt(np.mean((weights - optimum) ** 2))),
        "minimum_weight": float(weights.min()),
        "maximum_budget_error": float(np.max(abs(weights.sum(axis=1) - 1))),
    }


def run_experiment(output_dir, seed=7, n_assets=10, train_size=512,
                   test_size=128, epochs=150, turnover_limit=None, risk_aversion=5.0):
    if min(train_size, test_size, epochs) < 1 or n_assets < 2:
        raise ValueError("Use positive sample/epoch counts and at least two assets.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_mean, train_cov = sample_instances(train_size, n_assets, rng)
    test_mean, test_cov = sample_instances(test_size, n_assets, rng)
    train_previous = rng.dirichlet(np.ones(n_assets), train_size)
    test_previous = rng.dirichlet(np.ones(n_assets), test_size)
    solver = MeanVarianceSolver(n_assets, risk_aversion, turnover_limit)
    start = perf_counter()
    targets = np.array([solver.solve(m, c, p)
                        for m, c, p in zip(train_mean, train_cov, train_previous)])
    label_seconds = perf_counter() - start
    start = perf_counter()
    model, center, scale = fit_model(
        features(train_mean, train_cov, train_previous), targets, n_assets, epochs, turnover_limit
    )
    train_seconds = perf_counter() - start
    return _save_experiment(output_dir, model, center, scale, solver,
                            test_mean, test_cov, test_previous, label_seconds, train_seconds,
                            seed, train_size, epochs, risk_aversion)


def _save_experiment(output_dir, model, center, scale, solver, means, covariance, previous,
                     label_seconds, train_seconds, seed, train_size, epochs, risk_aversion):
    optimum = np.array([solver.solve(m, c, p) for m, c, p in zip(means, covariance, previous)])
    weights = predict(model, center, scale, means, covariance, previous, solver.turnover_limit)
    metrics = evaluate(weights, optimum, means, covariance, risk_aversion)
    metrics.update(benchmark(model, center, scale, means, covariance, previous, solver))
    metrics["turnover_limit"] = solver.turnover_limit
    metrics["maximum_turnover"] = float(np.abs(weights - previous).sum(axis=1).max())
    if solver.turnover_limit is not None:
        # Equal weight can violate a turnover cap, so its gap is not a feasible baseline.
        metrics.pop("equal_weight_objective_gap_mean")
    saving = (metrics["solver_single_median_ms"] - metrics["nn_single_median_ms"]) / 1000
    metrics.update({
        "label_seconds": label_seconds, "train_seconds": train_seconds,
        "break_even_instances": math.ceil((label_seconds + train_seconds) / saving) if saving > 0 else None,
        "seed": seed, "n_assets": means.shape[1], "train_size": train_size,
        "test_size": len(means), "epochs": epochs, "risk_aversion": risk_aversion,
        "interpretation": "Break-even ignores approximation error; review objective gaps separately.",
    })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "amortized_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "center": torch.tensor(center),
                "scale": torch.tensor(scale), "n_assets": means.shape[1],
                "turnover_limit": solver.turnover_limit},
               output_dir / "amortized_model.pt")
    return metrics
