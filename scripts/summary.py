# validation AP for every model (one memory build), then one clean table with test next to it
# python -u -m scripts.summary
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_transactions, split_data
from src.evaluate import evaluate
from src.features import current_matrix, artifact_matrix, profile_matrix, memory_matrix, memory_columns
from src.model import make_model

OUT = Path("docs/results")
VALID = OUT / "validation.csv"

# model name, readable name, memory per account
MODELS = [
    ("baseline_paper", "baseline", "none"),
    ("baseline_memory_noema_paper", "+ long-term", "O(1)"),
    ("baseline_memory_paper", "+ long-term + decay", "O(1)"),
    ("baseline_memory_noema_windows_paper", "+ long-term + window", "O(k)"),
    ("baseline_memory_windows_paper", "+ long-term + decay + window", "O(k)"),
    ("baseline_profile_paper", "baseline + profile", "none"),
    ("baseline_memory_profile_paper", "+ long-term + decay + profile", "O(1)"),
    ("baseline_memory_windows_profile_paper", "+ all memory + profile", "O(k)"),
]


def validation_scores():
    done = pd.read_csv(VALID) if VALID.exists() else pd.DataFrame(columns=["model"])
    todo = [m for m, _, _ in MODELS if m not in set(done["model"])]
    if not todo:
        return done

    df = load_transactions(Path("data/raw/tx.csv"))
    y = df["is_laundering"].to_numpy()
    train_end, valid_end = split_data(len(df))
    base = np.hstack([current_matrix(df), artifact_matrix(df)])
    profile = profile_matrix(df)
    memory = memory_matrix(df, "amount_received")
    blocks = {
        "baseline_paper": [base],
        "baseline_memory_noema_paper": [base, memory[:, memory_columns(False, ema=False)]],
        "baseline_memory_paper": [base, memory[:, memory_columns(False, ema=True)]],
        "baseline_memory_noema_windows_paper": [base, memory[:, memory_columns(True, ema=False)]],
        "baseline_memory_windows_paper": [base, memory[:, memory_columns(True, ema=True)]],
        "baseline_profile_paper": [base, profile],
        "baseline_memory_profile_paper": [base, profile, memory[:, memory_columns(False, ema=True)]],
        "baseline_memory_windows_profile_paper": [base, profile, memory[:, memory_columns(True, ema=True)]],
    }

    for name in todo:
        X = np.hstack(blocks[name])
        model = make_model(y[:train_end])
        model.fit(X[:train_end], y[:train_end])
        m = evaluate(y[train_end:valid_end], model.predict_proba(X[train_end:valid_end])[:, 1])
        row = pd.DataFrame([{
            "model": name,
            "valid_ap": round(m["average_precision"], 4),
            "valid_p_at_100": m["precision_at_100"],
        }])
        row.to_csv(VALID, mode="a", header=not VALID.exists(), index=False)   # saved right away
        print(f"{name:<40} valid AP {m['average_precision']:.4f}")
    return pd.read_csv(VALID)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    valid = validation_scores().set_index("model")
    test = (pd.read_csv("reports/final/test_per_seed.csv")
            .groupby("model")
            .agg(test_ap=("average_precision", "mean"),
                 test_ap_sd=("average_precision", "std"),
                 test_p_at_100=("precision_at_100", "mean")))

    rows = []
    for name, label, memory in MODELS:
        rows.append({
            "model": label,
            "memory_per_account": memory,
            "valid_ap": round(valid.loc[name, "valid_ap"], 3),
            "test_ap": round(test.loc[name, "test_ap"], 3),
            "test_ap_sd": round(test.loc[name, "test_ap_sd"], 3),
            "test_p_at_100": round(test.loc[name, "test_p_at_100"], 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))
