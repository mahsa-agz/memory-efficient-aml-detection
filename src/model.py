import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback


def make_model(y_train: np.ndarray, seed: int = 42) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    return XGBClassifier(
        objective="binary:logistic",   # the loss: log loss
        n_estimators=180,              # number of trees
        max_depth=5,                   # each tree asks at most 5 questions in a row
        learning_rate=0.08,            # each tree adds only 8% of its correction
        subsample=0.80,                # each tree sees a random 80% of the rows
        colsample_bytree=0.80,         # each tree sees a random 80% of the features
        min_child_weight=3,            # do not split on tiny groups of rows
        reg_lambda=2.0,                # penalty that keeps tree outputs small
        scale_pos_weight=negatives / positives,  # laundering rows count ~613x
        tree_method="hist",            # fast training on large data
        eval_metric="aucpr",           # track average precision while training
        n_jobs=-1,                     # use all CPU cores
        random_state=seed,
    )


class ProgressBar(TrainingCallback):
    """A tqdm bar that moves one step per tree."""

    def __init__(self, total: int):
        self.bar = tqdm(total=total, desc="trees")

    def after_iteration(self, model, epoch, evals_log):
        self.bar.update(1)
        return False          # False means "keep training"

    def after_training(self, model):
        self.bar.close()
        return model