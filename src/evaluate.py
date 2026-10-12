import numpy as np 
from pathlib import Path
from sklearn.metrics import average_precision_score
import csv
from datetime import datetime

def precision_at_k (y_true, score, k=100):
    y_true, score = np.asarray(y_true), np.asarray(score)
    top = np.argsort(score)[-k:]
    return float(y_true[top].mean())

def evaluate(y_true, score):
    return {
        "rows": len(y_true),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, score)),
        "precision_at_100": precision_at_k(y_true, score),
    }

import json
import platform
import subprocess

import pandas as pd
import sklearn
import xgboost



def save_result(name, metrics, model, feature_names, score_valid=None, folder="reports"):
    folder = Path(folder)
    runs = folder / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    run_id = f"{now:%Y%m%d_%H%M%S}_{name}"


    # 1. every detail, one JSON per run
    details = {
        "run_id": run_id,
        "model": name,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "n_features": len(feature_names),
        "features": feature_names,
        "params": model.get_params(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with (runs / f"{run_id}.json").open("w") as f:
        json.dump(details, f, indent=2, default=str)

    # 2. the validation scores, to recompute any metric later without retraining
    if score_valid is not None:
        np.save(runs / f"{run_id}_scores.npy", score_valid)

    # 3. one summary row in the CSV
    row = {"run_id": run_id, "model": name,
           "n_features": len(feature_names), **metrics}
    csv_path = folder / "results.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print("saved:", runs / f"{run_id}.json")