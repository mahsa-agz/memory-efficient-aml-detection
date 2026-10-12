"""Sanity checks. Run: python -m scripts.check"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_transactions, split_data
from src.evaluate import evaluate
from src.features import (current_matrix, memory_matrix, AccountState, similarity,
                          CURRENT_FEATURES, SIDE_FEATURES)


def check_split(frame):
    train_end, valid_end = split_data(len(frame))
    parts = {"train": frame.iloc[:train_end],
             "validation": frame.iloc[train_end:valid_end],
             "test": frame.iloc[valid_end:]}
    for name, part in parts.items():
        print(f"{name:<11} rows {len(part):>9,}  positives {int(part['is_laundering'].sum()):>5,}  "
              f"from {part['timestamp'].min()}  to {part['timestamp'].max()}")


def check_metric(frame):
    _, valid_end = split_data(len(frame))
    y_test = frame["is_laundering"].iloc[valid_end:].to_numpy()
    rng = np.random.default_rng(0)
    print("random :", evaluate(y_test, rng.random(len(y_test))))
    print("perfect:", evaluate(y_test, y_test.astype(float)))


def check_features(frame):
    X = current_matrix(frame)
    print("shape:", X.shape, " any NaN:", bool(np.isnan(X).any()))
    print(pd.DataFrame(X[:5], columns=CURRENT_FEATURES))


def check_memory():
    # three made-up transactions where the right answer is known by hand
    a = AccountState()
    for t, (x, cp) in enumerate([(10, "B"), (20, "C"), (60, "B")]):
        print(x, cp, np.round(a.features(x, t * 3600, cp), 3))
        a.update(x, t * 3600, cp, "out")
    # first row all zeros; third row: count 2, mean 15, counterparty_frequency 1
    print("std:", round((a.m2_amount / (a.count - 1)) ** 0.5, 2), "(expect 26.46)")
    print("self-similarity:", similarity(a, a), "(expect 1.0)")


def check_memory_rescan(frame, receiver_amount, n_rows=50_000, n_samples=200):
    # the real test of READ before WRITE: rebuild a sender's state from only its
    # earlier transactions, and compare with what memory_matrix produced
    small = frame.iloc[:n_rows].reset_index(drop=True)
    M = memory_matrix(small, receiver_amount)
    seconds = small["timestamp"].astype("int64").to_numpy() // 10**9
    senders = small["sender_account"].to_numpy()
    receivers = small["receiver_account"].to_numpy()
    paid = small["amount_paid"].to_numpy()
    r_amount = small[receiver_amount].to_numpy()

    rng = np.random.default_rng(0)
    failures = 0
    for i in rng.choice(n_rows, n_samples, replace=False):
        account = senders[i]
        state = AccountState()
        for j in range(i):                      # only rows BEFORE i
            if senders[j] == account:
                state.update(paid[j], seconds[j], receivers[j], "out")
            if receivers[j] == account:
                state.update(r_amount[j], seconds[j], senders[j], "in")
        expected = state.features(paid[i], seconds[i], receivers[i])
        got = M[i, :len(SIDE_FEATURES)]
        if not np.allclose(got, expected, rtol=1e-3, atol=1e-3):
            failures += 1

    print(f"[{receiver_amount}] memory NaN:", bool(np.isnan(M).any()))
    print(f"[{receiver_amount}] rescan: {n_samples - failures}/{n_samples} rows match",
          "PASS" if failures == 0 else "FAIL")
if __name__ == "__main__":
    check_memory()                              # no data needed, runs instantly

    frame = load_transactions(Path("data/raw/tx.csv"))
    check_split(frame)
    check_metric(frame)
    check_features(frame)
    check_memory_rescan(frame, "amount_received")   # paper protocol
    check_memory_rescan(frame, "amount_paid")       # clean protocol