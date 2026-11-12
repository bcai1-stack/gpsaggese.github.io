"""Rolling train/validation/test evaluation with non-overlapping holdings."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import make_samples
from .learning import train_model
from .metrics import hold_period, paired_utility_interval, summarize
from .optimization import MeanVarianceSolver


@dataclass
class Config:
    lookback: int = 126
    horizon: int = 21
    train_periods: int = 60
    validation_periods: int = 12
    test_periods: int = 12
    epochs: int = 15
    risk_aversion: float = 5.0
    cost_bps: float = 10.0
    seed: int = 7


def training_data(samples, start, end):
    return tuple(samples[key][start:end] for key in ("features", "factors", "future"))


def fit_fold(samples, first_test, config, output):
    validation_start = first_test - config.validation_periods
    train_start = validation_start - config.train_periods
    train = training_data(samples, train_start, validation_start)
    validation = training_data(samples, validation_start, first_test)
    fitted = {}
    for name, mode in (("predict_then_optimize", "mse"), ("decision_focused", "utility")):
        model = train_model(train, validation, mode, risk_aversion=config.risk_aversion,
                            epochs=config.epochs, seed=config.seed)
        fitted[name] = model
        torch.save({"state_dict": model.model.state_dict(),
                    "center": torch.tensor(model.center), "scale": torch.tensor(model.scale),
                    "metadata": model.metadata}, output / f"fold_{first_test}_{mode}.pt")
    return fitted


def allocate(samples, index, models, solver):
    factor = samples["factors"][index]
    covariance = factor.T @ factor
    mean = samples["means"][index]
    weights = {"equal_weight": np.full(len(mean), 1 / len(mean)),
               "minimum_variance": solver.solve(np.zeros_like(mean), covariance),
               "historical_mvo": solver.solve(mean, covariance),
               "shrunk_mean_mvo": solver.solve(0.25 * mean, covariance)}
    for name, model in models.items():
        predicted = model.predict_mean(samples["features"][index])
        weights[name] = solver.solve(predicted, covariance)
    return weights


def evaluate_fold(frame, samples, first, last, models, solver, previous, config):
    daily_rows, weight_rows = [], []
    for index in range(first, last):
        start = samples["starts"][index]
        dates = frame.index[start:start + config.horizon]
        for name, target in allocate(samples, index, models, solver).items():
            # A new strategy starts in cash. Subsequent trades use drifted weights.
            before = previous.get(name, np.zeros_like(target))
            daily, previous[name], turnover = hold_period(
                target, before, samples["future"][index], config.cost_bps)
            daily_rows.extend({"date": date, "strategy": name, "return": value}
                              for date, value in zip(dates, daily))
            row = {"date": dates[0], "strategy": name, "turnover": turnover}
            row.update(dict(zip(frame.columns, target)))
            weight_rows.append(row)
    return daily_rows, weight_rows


def fold_metadata(frame, samples, first, last, config, models):
    starts = samples["starts"]
    train_start = first - config.validation_periods - config.train_periods
    validation_start = first - config.validation_periods
    return {"training_first_label": str(frame.index[starts[train_start]].date()),
            "training_last_label": str(frame.index[starts[validation_start] - 1].date()),
            "validation_first_label": str(frame.index[starts[validation_start]].date()),
            "validation_last_label": str(frame.index[starts[first] - 1].date()),
            "test_start": str(frame.index[starts[first]].date()),
            "test_end": str(frame.index[starts[last - 1] + config.horizon - 1].date()),
            "models": {name: model.metadata for name, model in models.items()}}


def run_backtest(frame, output_dir, config=None, source="synthetic"):
    config = config or Config()
    if min(config.train_periods, config.validation_periods, config.test_periods, config.epochs) < 1:
        raise ValueError("Training, validation, test and epoch counts must be positive.")
    if not np.isfinite(config.cost_bps) or config.cost_bps < 0:
        raise ValueError("Cost in bps must be finite and nonnegative.")
    samples = make_samples(frame, config.lookback, config.horizon)
    first_test = config.train_periods + config.validation_periods
    if first_test >= len(samples["starts"]):
        raise ValueError("Not enough complete periods for train, validation and test.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    solver = MeanVarianceSolver(frame.shape[1], config.risk_aversion)
    previous, daily_rows, weight_rows, folds = {}, [], [], []
    for first in range(first_test, len(samples["starts"]), config.test_periods):
        last = min(first + config.test_periods, len(samples["starts"]))
        print(f"Training fold {len(folds) + 1}, test {frame.index[samples['starts'][first]].date()}", flush=True)
        models = fit_fold(samples, first, config, output)
        daily, weights = evaluate_fold(frame, samples, first, last, models, solver, previous, config)
        daily_rows.extend(daily)
        weight_rows.extend(weights)
        folds.append(fold_metadata(frame, samples, first, last, config, models))
    return save_results(frame, output, config, source, daily_rows, weight_rows, folds)


def save_results(frame, output, config, source, daily_rows, weight_rows, folds):
    daily = pd.DataFrame(daily_rows).pivot(index="date", columns="strategy", values="return")
    weights = pd.DataFrame(weight_rows)
    metrics = {name: summarize(daily[name], weights.loc[weights.strategy == name, "turnover"],
                               config.risk_aversion) for name in daily.columns}
    for name in daily.columns:
        holdings = weights.loc[weights.strategy == name, list(frame.columns)]
        metrics[name]["maximum_target_weight"] = float(holdings.to_numpy().max())
    intervals = {name: paired_utility_interval(daily.decision_focused, daily[name],
                                              config.risk_aversion, config.seed)
                 for name in ("equal_weight", "minimum_variance", "predict_then_optimize")}
    report = {"config": asdict(config), "source": source, "assets": list(frame.columns),
              "input_sha256": hashlib.sha256(frame.to_csv().encode()).hexdigest(),
              "test_start": str(daily.index[0].date()), "test_end": str(daily.index[-1].date()),
              "test_days": len(daily), "metrics": metrics,
              "decision_focused_minus_baseline_utility_ci": intervals, "folds": folds}
    daily.to_csv(output / "daily_returns.csv")
    weights.to_csv(output / "weights.csv", index=False)
    pd.DataFrame(metrics).T.to_csv(output / "summary.csv", index_label="strategy")
    (output / "backtest_report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_results(daily, output)
    return report


def plot_results(daily, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wealth = pd.concat([pd.DataFrame(1., index=[daily.index[0] - pd.Timedelta(days=1)],
                                    columns=daily.columns), (1 + daily).cumprod()])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    wealth.plot(ax=axes[0], ylabel="Net wealth (initial = 1)")
    (wealth / wealth.cummax() - 1).plot(ax=axes[1], ylabel="Drawdown", legend=False)
    fig.tight_layout()
    fig.savefig(output / "backtest.png", dpi=160)
    plt.close(fig)
