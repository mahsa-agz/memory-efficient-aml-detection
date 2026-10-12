"""
Random search for XGBoost settings, done the same way for baseline and memory model.
Run: python -m scripts.tune paper     (or clean)

Every choice here is made on VALIDATION. Test is not touched.
"""
import csv
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.data import load_transactions, split_data
from src.features import current_matrix, artifact_matrix, memory_matrix, PROTOCOLS
from src.model import make_model


N_TRIALS = 20
CACHE = Path("data/processed")
PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else "paper"
OUT = Path(f"reports/tuning_{PROTOCOL}.csv")


def cached_memory(df, protocol):
    # the slow part; cached per protocol, with a fingerprint of features.py in the name
    version = zlib.crc32(Path("src/features.py").read_bytes())
    path = CACHE / f"memory_{protocol}_{version:08x}.npy"
    if path.exists():
        print("loading cache:", path)
        return np.load(path)
    memory = memory_matrix(df, PROTOCOLS[protocol])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, memory)
    print("saved cache:", path)
    return memory


def sample_params(rng):
    return {
        "max_depth": int(rng.integers(3, 9)),
        "min_child_weight": float(rng.choice([1, 3, 10, 30])),
        "subsample": round(float(rng.uniform(0.6, 1.0)), 2),
        "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 2),
        "reg_lambda": float(rng.choice([0.5, 1, 2, 5, 10])),
    }


def run_trial(params, X_train, y_train, X_valid, y_valid):
    model = make_model(y_train)
    model.set_params(n_estimators=2000, learning_rate=0.05, early_stopping_rounds=50, **params)
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    score = model.predict_proba(X_valid)[:, 1]
    return {
        **params,
        "trees": model.best_iteration + 1,
        "valid_ap": float(average_precision_score(y_valid, score)),
        "seconds": round(time.perf_counter() - start),
    }


def save_row(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    with OUT.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    if PROTOCOL not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {list(PROTOCOLS)}, got {PROTOCOL!r}")

    df = load_transactions(Path("data/raw/tx.csv"))
    y = df["is_laundering"].to_numpy()
    train_end, valid_end = split_data(len(df))

    base = current_matrix(df)
    if PROTOCOL == "paper":
        base = np.hstack([base, artifact_matrix(df)])
    memory = cached_memory(df, PROTOCOL)

    feature_sets = {
        f"baseline_{PROTOCOL}": base,
        f"memory_{PROTOCOL}": np.hstack([base, memory]),
    }

    # the SAME 20 settings for both models, so neither gets luckier draws
    rng = np.random.default_rng(0)
    trials = [sample_params(rng) for _ in range(N_TRIALS)]

    for name, X in feature_sets.items():
        best = None
        for i, params in enumerate(trials):
            result = run_trial(params, X[:train_end], y[:train_end],
                               X[train_end:valid_end], y[train_end:valid_end])
            row = {"model": name, "trial": i, **result}
            save_row(row)
            print(f"{name:<16} trial {i:>2}  valid AP {result['valid_ap']:.4f}  "
                  f"trees {result['trees']:>4}  {result['seconds']} s")
            if best is None or result["valid_ap"] > best["valid_ap"]:
                best = row
        print("BEST", best)