"""Daily return inputs and strictly historical portfolio features."""

from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .optimization import covariance_factor

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/"
    "10_Industry_Portfolios_daily_CSV.zip"
)


def validate_returns(frame):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("Provide at least two assets and nonempty daily returns.")
    if frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError("Dates must be valid and unique.")
    if not frame.index.is_monotonic_increasing or frame.columns.has_duplicates:
        raise ValueError("Dates must be sorted and asset names unique.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Returns must be finite decimals greater than -1.")
    frame.index.name = "date"
    return frame.astype(float)


def parse_french(payload):
    with ZipFile(BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8-sig")
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns" in line)
    header = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith(","))
    rows = []
    for line in lines[header + 1:]:
        if not re.match(r"^\s*\d{8},", line):
            if rows:
                break
            continue
        rows.append(line)
    # The archive also contains equal-weighted returns. Read only its first table.
    frame = pd.read_csv(StringIO("\n".join([lines[header], *rows])), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame.columns = frame.columns.str.strip()
    frame = frame.replace([-99.99, -999.0], np.nan) / 100
    return frame


def download_french(destination, start="2010-01-01", end="2025-12-31"):
    with urlopen(FRENCH_URL, timeout=60) as response:
        payload = response.read()
    frame = validate_returns(parse_french(payload).loc[start:end])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination)
    metadata = {"source": FRENCH_URL, "start": str(frame.index[0].date()),
                "end": str(frame.index[-1].date()), "rows": len(frame),
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "csv_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return frame


def synthetic_returns(days=2200, assets=10, seed=7):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.008, (days, 1))
    loading = rng.uniform(0.5, 1.2, (1, assets))
    noise = rng.normal(0, 0.008, (days, assets))
    values = market @ loading + noise + 0.0001
    return validate_returns(pd.DataFrame(values, index=pd.bdate_range("2010-01-01", periods=days),
                                         columns=[f"asset_{i}" for i in range(assets)]))


def make_samples(frame, lookback=126, horizon=21):
    if lookback < 63 or horizon < 2:
        raise ValueError("lookback must be >= 63 and horizon >= 2.")
    values = validate_returns(frame).to_numpy()
    starts = np.arange(lookback, len(values) - horizon + 1, horizon)
    features, factors, means, future = [], [], [], []
    for start in starts:
        # The allocation at start can only see returns strictly before start.
        history = values[start - lookback:start]
        feature = [history[-days:].mean(0) * 252 for days in (5, 21, 63)]
        features.append(np.concatenate([*feature, history.std(0) * np.sqrt(252)]))
        covariance = LedoitWolf().fit(history).covariance_ * 252
        factors.append(covariance_factor(covariance))
        means.append(history.mean(0) * 252)
        future.append(values[start:start + horizon])
    if not len(starts):
        raise ValueError("Insufficient observations for one complete feature/return window.")
    return {"features": np.array(features), "factors": np.array(factors),
            "means": np.array(means), "future": np.array(future), "starts": starts}
