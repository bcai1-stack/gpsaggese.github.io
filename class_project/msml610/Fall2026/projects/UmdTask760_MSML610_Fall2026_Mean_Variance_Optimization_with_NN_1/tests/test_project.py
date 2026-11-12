"""Check optimization, gradients, data parsing and complete evaluation paths."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from mvo.amortized import constrain_turnover, run_experiment
from mvo.backtest import Config, run_backtest
from mvo.data import parse_french, synthetic_returns
from mvo.optimization import MeanVarianceSolver, covariance_factor, make_allocation_layer


class ProjectTests(unittest.TestCase):
    def test_covariance_validation(self):
        for covariance in ([[1, 2], [0, 1]], [[1, 0], [0, -1]], [[np.nan]]):
            with self.assertRaises(ValueError):
                covariance_factor(covariance)
        singular = np.ones((3, 3))
        factor = covariance_factor(singular)
        np.testing.assert_allclose(factor.T @ factor, singular, atol=1e-12)

    def test_turnover_constraint_binds(self):
        previous = np.array([0.5, 0.5])
        covariance = np.eye(2) * 0.01
        mean = np.array([1.0, -1.0])
        solver = MeanVarianceSolver(2, turnover_limit=0.2)
        weights = solver.solve(mean, covariance, previous)
        np.testing.assert_allclose(weights, [0.6, 0.4], atol=1e-5)
        with self.assertRaises(ValueError):
            solver.solve(mean, covariance)
        for limit in (0, -0.1, 2.1, np.nan):
            with self.assertRaises(ValueError):
                MeanVarianceSolver(2, turnover_limit=limit)

    def test_gradient_matches_finite_difference(self):
        covariance = np.diag([0.04, 0.09])
        mean = torch.tensor([0.03, 0.04], dtype=torch.float64, requires_grad=True)
        factor = torch.tensor(covariance_factor(covariance))
        layer = make_allocation_layer(2)
        weights, = layer(mean, factor, solver_args={"eps": 1e-8})
        weights[0].backward()
        solver = MeanVarianceSolver(2)
        epsilon = 1e-4
        numeric = []
        for direction in np.eye(2) * epsilon:
            upper = solver.solve(mean.detach().numpy() + direction, covariance)[0]
            lower = solver.solve(mean.detach().numpy() - direction, covariance)[0]
            numeric.append((upper - lower) / (2 * epsilon))
        np.testing.assert_allclose(mean.grad.numpy(), numeric, atol=2e-3, rtol=2e-3)

    def test_constrained_layer_matches_solver(self):
        mean = torch.tensor([0.8, -0.2], dtype=torch.float64, requires_grad=True)
        covariance = np.eye(2) * 0.05
        previous = torch.tensor([0.4, 0.6], dtype=torch.float64)
        factor = torch.tensor(covariance_factor(covariance))
        weights, = make_allocation_layer(2, turnover_limit=0.3)(
            mean, factor, previous, solver_args={"eps": 1e-8}
        )
        expected = MeanVarianceSolver(2, turnover_limit=0.3).solve(
            mean.detach().numpy(), covariance, previous.numpy()
        )
        np.testing.assert_allclose(weights.detach().numpy(), expected, atol=2e-4)

    def test_neural_turnover_projection(self):
        previous = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
        proposed = torch.tensor([[0.9, 0.1], [0.5, 0.5]], requires_grad=True)
        result = constrain_turnover(proposed, previous, 0.2)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result - previous).abs().sum(1) <= 0.200001).all())
        np.testing.assert_allclose(result.detach().numpy().sum(1), 1)
        result[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(proposed.grad).all())

    def test_archive_parser_selects_value_weighted_table(self):
        text = ("Average Value Weighted Returns -- Daily\n,A,B\n"
                "20200102,1.0,2.0\n20200103,-99.99,0.0\n\n"
                "Average Equal Weighted Returns -- Daily\n,A,B\n20200102,9,9\n")
        payload = BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("returns.csv", text)
        frame = parse_french(payload.getvalue())
        self.assertEqual(frame.shape, (2, 2))
        self.assertAlmostEqual(frame.iloc[0, 0], 0.01)
        self.assertTrue(np.isnan(frame.iloc[1, 0]))

    def test_amortized_end_to_end(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as output:
            report = run_experiment(output, n_assets=3, train_size=16,
                                    test_size=8, epochs=2, turnover_limit=0.4)
            self.assertLessEqual(report["maximum_turnover"], 0.40001)
            self.assertLess(report["maximum_budget_error"], 1e-6)
            self.assertGreaterEqual(report["objective_gap_mean"], -1e-6)
            self.assertTrue((Path(output) / "amortized_model.pt").exists())

    def test_walk_forward_end_to_end(self):
        torch.set_num_threads(1)
        frame = synthetic_returns(days=240, assets=3)
        config = Config(lookback=63, horizon=10, train_periods=5,
                        validation_periods=2, test_periods=3, epochs=1)
        with tempfile.TemporaryDirectory() as output:
            report = run_backtest(frame, output, config)
            daily = pd.read_csv(Path(output) / "daily_returns.csv", index_col=0)
            weights = pd.read_csv(Path(output) / "weights.csv")
            self.assertEqual(daily.shape[1], 6)
            self.assertEqual(len(daily), report["test_days"])
            self.assertTrue(np.isfinite(daily.to_numpy()).all())
            self.assertFalse(daily.index.duplicated().any())
            np.testing.assert_allclose(weights[list(frame.columns)].sum(axis=1), 1)
            for fold in report["folds"]:
                self.assertLess(fold["training_last_label"], fold["validation_first_label"])
                self.assertLess(fold["validation_last_label"], fold["test_start"])
            self.assertTrue((Path(output) / "backtest.png").exists())


if __name__ == "__main__":
    unittest.main()
