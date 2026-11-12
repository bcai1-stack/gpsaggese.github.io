"""Guard against look-ahead, incorrect turnover and broken solver gradients."""

import unittest

import numpy as np
import torch

from mvo.data import make_samples, synthetic_returns, validate_returns
from mvo.learning import buy_and_hold_returns, train_model
from mvo.metrics import hold_period, summarize
from mvo.optimization import MeanVarianceSolver, covariance_factor, make_allocation_layer


class Invariants(unittest.TestCase):
    def test_future_changes_do_not_change_current_features(self):
        frame = synthetic_returns(days=250, assets=3)
        original = make_samples(frame)
        frame.iloc[126:] += 0.01
        modified = make_samples(frame)
        for key in ("features", "factors", "means"):
            np.testing.assert_allclose(original[key][0], modified[key][0])
        self.assertFalse(np.allclose(original["future"][0], modified["future"][0]))

    def test_drift_cost_and_initial_cash(self):
        returns = np.array([[0.1, 0.0], [0.0, 0.2]])
        daily, drifted, turnover = hold_period(np.array([0.5, 0.5]), np.zeros(2), returns, 10)
        self.assertAlmostEqual(turnover, 1)
        self.assertAlmostEqual(np.prod(1 + daily), 0.999 * 1.15)
        np.testing.assert_allclose(drifted, [0.55 / 1.15, 0.6 / 1.15])
        _, _, next_turnover = hold_period(np.array([0.5, 0.5]), drifted, returns, 0)
        self.assertAlmostEqual(next_turnover, np.abs(0.5 - drifted).sum())

    def test_differentiable_holding_matches_accounting(self):
        weights = np.array([0.3, 0.7])
        returns = np.array([[0.1, -0.1], [-0.05, 0.2]])
        actual = buy_and_hold_returns(torch.tensor(weights[None]), torch.tensor(returns[None]))
        expected, _, _ = hold_period(weights, weights, returns, 0)
        np.testing.assert_allclose(actual.numpy()[0], expected)

    def test_minimum_variance_matches_analytic_solution(self):
        covariance = np.diag([0.04, 0.09])
        weights = MeanVarianceSolver(2).solve(np.zeros(2), covariance)
        np.testing.assert_allclose(weights, [9 / 13, 4 / 13], atol=1e-5)
        factor = covariance_factor(covariance)
        np.testing.assert_allclose(factor.T @ factor, covariance)

    def test_layer_has_nonzero_finite_gradient(self):
        mean = torch.tensor([0.03, 0.04], dtype=torch.float64, requires_grad=True)
        factor = torch.tensor(covariance_factor(np.diag([0.04, 0.09])))
        weights, = make_allocation_layer(2)(mean, factor)
        weights[0].backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertGreater(mean.grad.abs().sum().item(), 1e-4)

    def test_scaler_uses_training_only(self):
        samples = make_samples(synthetic_returns(days=350, assets=3))
        train = tuple(samples[key][:5] for key in ("features", "factors", "future"))
        validation = tuple(samples[key][5:] for key in ("features", "factors", "future"))
        fitted = train_model(train, validation, "mse", epochs=1)
        np.testing.assert_allclose(fitted.center, train[0].mean(0))

    def test_invalid_returns_and_first_day_drawdown(self):
        frame = synthetic_returns(days=100, assets=3)
        frame.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_returns(frame)
        metrics = summarize([-0.1, 0.0], [0])
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)


if __name__ == "__main__":
    unittest.main()
