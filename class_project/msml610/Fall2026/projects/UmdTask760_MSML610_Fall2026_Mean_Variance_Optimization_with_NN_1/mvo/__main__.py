"""Run with python -m mvo download, amortized, or backtest."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .amortized import run_experiment
from .backtest import Config, run_backtest
from .data import download_french, synthetic_returns, validate_returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--output", default="data/french10.csv")
    download.add_argument("--start", default="2010-01-01")
    download.add_argument("--end", default="2025-12-31")
    amortized = commands.add_parser("amortized")
    amortized.add_argument("--output", default="results/amortized")
    amortized.add_argument("--epochs", type=int, default=150)
    amortized.add_argument("--train-size", type=int, default=512)
    amortized.add_argument("--seed", type=int, default=7)
    amortized.add_argument("--turnover-limit", type=float, default=None)
    backtest = commands.add_parser("backtest")
    backtest.add_argument("--csv", type=Path, help="date index + decimal daily returns")
    backtest.add_argument("--output", default="results/backtest")
    backtest.add_argument("--epochs", type=int, default=15)
    backtest.add_argument("--cost-bps", type=float, default=10.)
    backtest.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "download":
        frame = download_french(args.output, args.start, args.end)
        print(f"Saved {len(frame)} rows to {args.output}")
    elif args.command == "amortized":
        print(json.dumps(run_experiment(args.output, seed=args.seed, epochs=args.epochs,
                                       train_size=args.train_size, turnover_limit=args.turnover_limit), indent=2))
    else:
        frame = synthetic_returns(seed=args.seed)
        if args.csv:
            frame = validate_returns(pd.read_csv(args.csv, index_col=0, parse_dates=True))
        config = Config(epochs=args.epochs, cost_bps=args.cost_bps, seed=args.seed)
        report = run_backtest(frame, args.output, config, str(args.csv) if args.csv else "synthetic")
        print(pd.DataFrame(report["metrics"]).T.round(4).to_string())


if __name__ == "__main__":
    main()
