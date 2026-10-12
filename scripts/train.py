"""
Train one model and score it on validation.
Run: python -m scripts.train <protocol> [memory] [noema] [windows] [profile]

    python -m scripts.train paper                            baseline
    python -m scripts.train paper memory noema               + long-term memory only
    python -m scripts.train paper memory                     + long-term + EMA      (compact)
    python -m scripts.train paper memory noema windows       + long-term + window
    python -m scripts.train paper memory windows             + long-term + EMA + window (full)
    add "profile" to any of them for the account profile
"""
import sys
import time
from pathlib import Path

from src.data import load_transactions, split_data
from src.evaluate import evaluate, save_result
from src.features import build_features
from src.model import make_model, ProgressBar

args = sys.argv[1:]
PROTOCOL = args[0] if args else "paper"
MEMORY = "memory" in args
WINDOWS = MEMORY and "windows" in args
EMA = not (MEMORY and "noema" in args)
PROFILE = "profile" in args
NAME = "_".join(["baseline"] + ["memory"] * MEMORY + ["noema"] * (not EMA)
                + ["windows"] * WINDOWS + ["profile"] * PROFILE + [PROTOCOL])

# 1. Data and features
df = load_transactions(Path("data/raw/tx.csv"))
X, names = build_features(df, PROTOCOL, memory=MEMORY, profile=PROFILE, windows=WINDOWS, ema=EMA)
y = df["is_laundering"].to_numpy()
print(f"{NAME}: {len(names)} columns")

# 2. Chronological split
train_end, valid_end = split_data(len(df))
X_train, y_train = X[:train_end], y[:train_end]
X_valid, y_valid = X[train_end:valid_end], y[train_end:valid_end]

# 3. Train
model = make_model(y_train)
model.set_params(callbacks=[ProgressBar(model.n_estimators)]) # type: ignore

start = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
fit_seconds = time.perf_counter() - start

# 4. Score validation only; test stays untouched until the end
score_valid = model.predict_proba(X_valid)[:, 1]
metrics = evaluate(y_valid, score_valid)
metrics["fit_seconds"] = round(fit_seconds)
print(metrics)
save_result(NAME, metrics, model, names, score_valid)

# 5. Save the model
Path("models").mkdir(exist_ok=True)
model.save_model(f"models/{NAME}.json")