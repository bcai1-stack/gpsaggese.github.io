"""Long-only mean–variance allocation shared by all experiments."""

import cvxpy as cp
import numpy as np


def covariance_factor(covariance):
    """Return F with F.T @ F = covariance, allowing roundoff at zero."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance must be a square matrix.")
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance must be finite.")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8):
        raise ValueError("Covariance must be symmetric.")
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if values.min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")
    return np.sqrt(np.maximum(values, 0))[:, None] * vectors.T


def clean_weights(weights, tolerance=2e-4):
    """Remove solver roundoff only after checking the original solution."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("Portfolio weights must be a finite vector.")
    if weights.min() < -tolerance or abs(weights.sum() - 1) > tolerance:
        raise ValueError("Solver returned infeasible portfolio weights.")
    weights = np.maximum(weights, 0)
    return weights / weights.sum()


def _allocation_problem(n_assets, risk_aversion, turnover_limit=None):
    if n_assets < 2 or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Use at least two assets and positive risk aversion.")
    mean = cp.Parameter(n_assets)
    factor = cp.Parameter((n_assets, n_assets))
    weights = cp.Variable(n_assets)
    previous = cp.Parameter(n_assets)
    # A covariance factor makes the problem DPP-compatible for differentiation.
    loss = risk_aversion * cp.sum_squares(factor @ weights) - mean @ weights
    constraints = [weights >= 0, cp.sum(weights) == 1]
    if turnover_limit is not None:
        if not np.isfinite(turnover_limit) or not 0 < turnover_limit <= 2:
            raise ValueError("Turnover limit must be in (0, 2].")
        constraints.append(cp.norm1(weights - previous) <= turnover_limit)
    problem = cp.Problem(cp.Minimize(loss), constraints)
    return problem, mean, factor, weights, previous


class MeanVarianceSolver:
    """Reuse one compiled convex problem across annualized input moments."""

    def __init__(self, n_assets, risk_aversion=5.0, turnover_limit=None):
        self.problem, self.mean, self.factor, self.weights, self.previous = _allocation_problem(
            n_assets, risk_aversion, turnover_limit
        )
        self.turnover_limit = turnover_limit

    def solve(self, mean, covariance, previous=None):
        mean = np.asarray(mean, dtype=float)
        if mean.shape != self.mean.shape or not np.isfinite(mean).all():
            raise ValueError("Expected returns have an invalid shape or value.")
        self.mean.value = mean
        self.factor.value = covariance_factor(covariance)
        if self.turnover_limit is not None:
            if previous is None:
                raise ValueError("Provide previous invested weights for a turnover constraint.")
            previous = clean_weights(previous)
            self.previous.value = previous
        self.problem.solve(solver=cp.CLARABEL, warm_start=True)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Allocation failed: {self.problem.status}")
        weights = clean_weights(self.weights.value)
        if self.turnover_limit is not None:
            if np.abs(weights - previous).sum() > self.turnover_limit + 2e-4:
                raise RuntimeError("Solver exceeded the turnover constraint.")
        return weights


def make_allocation_layer(n_assets, risk_aversion=5.0, turnover_limit=None):
    """Create a PyTorch layer accepting (annual_mean, covariance_factor)."""
    from cvxpylayers.torch import CvxpyLayer

    problem, mean, factor, weights, previous = _allocation_problem(
        n_assets, risk_aversion, turnover_limit
    )
    parameters = [mean, factor]
    if turnover_limit is not None:
        parameters.append(previous)
    return CvxpyLayer(problem, parameters=parameters, variables=[weights])
