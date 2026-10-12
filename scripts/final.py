"""
Final numbers: the test set, fixed settings, 3 seeds, plus the cost of the memory.
Run ONCE, with nothing else running: python -m scripts.final

Test is used here and nowhere else, and nothing is chosen from these numbers.
Models are trained on train only, the same as in every validation run.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_transactions, split_data
from src.evaluate import evaluate
from src.features import (current_matrix, artifact_matrix, profile_matrix, memory_matrix,
                          memory_columns, AccountState, similarity, PROTOCOLS)
from src.model import make_model

SEEDS = [0, 1, 2]
N_TIMED = 100_000           # transactions timed one by one for latency
OUT = Path("reports/final")


def test_runs(name, X, y, train_end, valid_end):
    # same fixed settings for every model; only the seed changes
    rows = []
    for seed in SEEDS:
        model = make_model(y[:train_end], seed=seed)
        model.fit(X[:train_end], y[:train_end])
        metrics = evaluate(y[valid_end:], model.predict_proba(X[valid_end:])[:, 1])
        rows.append({"model": name, "seed": seed, "n_features": X.shape[1], **metrics})
        print(f"{name:<24} seed {seed}  test AP {metrics['average_precision']:.4f}  "
              f"P@100 {metrics['precision_at_100']:.2f}")
    return rows, model


def window_bytes(recent):
    # the buffer of the last 20 transactions, counted deep: deque + each tuple + its items
    return sys.getsizeof(recent) + sum(sys.getsizeof(t) + sum(sys.getsizeof(x) for x in t)
                                       for t in recent)


EMA_FIELDS = ("ema_fast", "ema_slow")

# memory variant -> (keeps EMA, keeps the window buffer)
VARIANTS = {
    "long": (False, False),
    "long_ema": (True, False),
    "long_window": (False, True),
    "long_ema_window": (True, True),
}


def state_bytes(state, ema, windows):
    # Python's own sizes, so an upper bound: a compact implementation would be smaller.
    # sys.getsizeof of a numpy array that owns its data already includes the data.
    summary = sys.getsizeof(state) + sum(
        sys.getsizeof(v) for k, v in vars(state).items()
        if k != "recent" and (ema or k not in EMA_FIELDS))
    return summary + (window_bytes(state.recent) if windows else 0)


def memory_cost(df):
    # replay the whole stream as in the paper protocol; time the last N_TIMED transactions
    seconds = df["timestamp"].astype("int64").to_numpy() // 10**9
    senders = df["sender_account"].to_numpy()
    receivers = df["receiver_account"].to_numpy()
    paid = df["amount_paid"].to_numpy()
    received = df["amount_received"].to_numpy()
    n = len(df)
    first_timed = n - N_TIMED
    times = np.empty(N_TIMED)
    states = {}

    for i in range(n):
        t0 = time.perf_counter_ns()
        s = states.setdefault(senders[i], AccountState())
        r = states.setdefault(receivers[i], AccountState())
        s.features(paid[i], seconds[i], receivers[i])
        r.features(received[i], seconds[i], senders[i])
        similarity(s, r)
        s.update(paid[i], seconds[i], receivers[i], "out")
        r.update(received[i], seconds[i], senders[i], "in")
        if i >= first_timed:
            times[i - first_timed] = time.perf_counter_ns() - t0

    counts = pd.Series({acct: st.count for acct, st in states.items()})
    busiest = counts.idxmax()
    cost = {
        "feature_latency_median_us": round(float(np.median(times)) / 1000, 1),
        "feature_latency_p99_us": round(float(np.percentile(times, 99)) / 1000, 1),
        "accounts": int(len(states)),
        "busiest_account_transactions": int(counts[busiest]),
    }
    # memory per account for each variant of the memory
    for label, (ema, windows) in VARIANTS.items():
        sizes = pd.Series({acct: state_bytes(st, ema, windows) for acct, st in states.items()})
        cost[f"{label}_bytes_per_account_mean"] = int(sizes.mean())
        cost[f"{label}_bytes_per_account_max"] = int(sizes.max())
        cost[f"{label}_busiest_account_bytes"] = int(sizes[busiest])
        cost[f"{label}_total_state_mb"] = round(float(sizes.sum()) / 1e6, 1)
    return cost


def model_cost(model, X_test):
    # one transaction at a time, as in a live system; then the whole test set at once
    booster = model.get_booster()
    rows = X_test[:2000]
    times = np.empty(len(rows))
    for k in range(len(rows)):
        t0 = time.perf_counter_ns()
        booster.inplace_predict(rows[k:k + 1])
        times[k] = time.perf_counter_ns() - t0
    t0 = time.perf_counter()
    model.predict_proba(X_test)
    batch_seconds = time.perf_counter() - t0
    return {
        "model_latency_median_us": round(float(np.median(times)) / 1000, 1),
        "model_latency_p99_us": round(float(np.percentile(times, 99)) / 1000, 1),
        "batch_scoring_tx_per_s": int(len(X_test) / batch_seconds),
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_transactions(Path("data/raw/tx.csv"))
    y = df["is_laundering"].to_numpy()
    train_end, valid_end = split_data(len(df))

    # every block once, then combined
    current = current_matrix(df)
    artifact = artifact_matrix(df)
    profile = profile_matrix(df)
    memory_all = memory_matrix(df, PROTOCOLS["paper"])
    long = memory_all[:, memory_columns(windows=False, ema=False)]
    long_ema = memory_all[:, memory_columns(windows=False, ema=True)]
    long_window = memory_all[:, memory_columns(windows=True, ema=False)]
    long_ema_window = memory_all[:, memory_columns(windows=True, ema=True)]
    base = np.hstack([current, artifact])

    # model -> (features, memory variant for the bytes column); paper protocol only for now
    models = {
        "baseline_paper": (base, None),
        "baseline_memory_noema_paper": (np.hstack([base, long]), "long"),
        "baseline_memory_paper": (np.hstack([base, long_ema]), "long_ema"),
        "baseline_memory_noema_windows_paper": (np.hstack([base, long_window]), "long_window"),
        "baseline_memory_windows_paper": (np.hstack([base, long_ema_window]), "long_ema_window"),
        "baseline_profile_paper": (np.hstack([base, profile]), None),
        "baseline_memory_profile_paper": (np.hstack([base, profile, long_ema]), "long_ema"),
        "baseline_memory_windows_profile_paper": (np.hstack([base, profile, long_ema_window]),
                                                  "long_ema_window"),
    }

    # models already scored on test are kept, not rerun: their columns have not changed
    done = pd.read_csv(OUT / "test_per_seed.csv") if (OUT / "test_per_seed.csv").exists() else pd.DataFrame()
    all_rows = done.to_dict("records")
    last_memory_model = None
    for name, (X, _) in models.items():
        if not done.empty and name in set(done["model"]):
            print(f"{name:<38} already on test, kept")
            continue
        rows, model = test_runs(name, X, y, train_end, valid_end)
        all_rows += rows
        if name == "baseline_memory_windows_paper":
            last_memory_model, last_X = model, X
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(OUT / "test_per_seed.csv", index=False)

    summary = (per_seed.groupby("model", sort=False)
               .agg(n_features=("n_features", "first"),
                    ap_mean=("average_precision", "mean"),
                    ap_sd=("average_precision", "std"),
                    p_at_100=("precision_at_100", "mean"),
                    prevalence=("prevalence", "first"))
               .round(4))

    print("\nmeasuring cost (nothing else should be running) ...")
    cost = {**memory_cost(df), **model_cost(last_memory_model, last_X[valid_end:])}
    with (OUT / "cost.json").open("w") as f:
        json.dump(cost, f, indent=2)

    # memory per account next to each model's AP: the trade-off in one table
    summary["bytes_per_account"] = [
        cost[f"{models[name][1]}_bytes_per_account_max"] if name in models and models[name][1] else 0
        for name in summary.index]
    summary.to_csv(OUT / "test_summary.csv")

    print("\nTEST, 3 seeds, fixed settings")
    print(summary.to_string())
    print("\nCOST")
    for k, v in cost.items():
        print(f"  {k:<32} {v}")
