"""Portfolio accounting and paired out-of-sample comparisons."""

import numpy as np


def hold_period(target, previous, returns, cost_bps):
    """Charge cost per dollar bought or sold, then let holdings drift."""
    if cost_bps < 0:
        raise ValueError("Trading cost must be nonnegative.")
    turnover = float(np.abs(target - previous).sum())
    cost = cost_bps * 1e-4 * turnover
    if cost >= 1:
        raise ValueError("Trading costs consume the entire portfolio.")
    holdings = np.asarray(target, dtype=float).copy()
    daily = []
    for row in returns:
        growth = float(holdings @ (1 + row))
        daily.append(growth - 1)
        holdings = holdings * (1 + row) / growth
    # Costs are paid before the first day's market move.
    daily[0] = (1 - cost) * (1 + daily[0]) - 1
    return np.array(daily), holdings, turnover


def summarize(daily, turnover, risk_aversion=5.0):
    values = np.asarray(daily)
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    variance = float(values.var(ddof=0)) * 252
    volatility = np.sqrt(variance)
    mean = float(values.mean()) * 252
    return {"cagr": float(wealth[-1] ** (252 / len(values)) - 1),
            "annual_volatility": float(volatility),
            "sharpe_rf_zero": float(mean / volatility) if volatility > 0 else None,
            "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
            "annual_utility": mean - risk_aversion * variance,
            "annual_turnover": float(np.sum(turnover) * 252 / len(values))}


def paired_utility_interval(candidate, baseline, risk_aversion=5.0, seed=7,
                            block=21, draws=1000):
    """Circular block bootstrap keeps paired dates and short-run dependence."""
    rng = np.random.default_rng(seed)
    count = len(candidate)
    differences = []
    for _ in range(draws):
        starts = rng.integers(0, count, size=int(np.ceil(count / block)))
        indices = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        left, right = np.asarray(candidate)[indices], np.asarray(baseline)[indices]
        delta = left.mean() - right.mean() - risk_aversion * (left.var() - right.var())
        differences.append(252 * delta)
    return {"lower_95": float(np.quantile(differences, 0.025)),
            "upper_95": float(np.quantile(differences, 0.975)),
            "block_days": block, "draws": draws}
