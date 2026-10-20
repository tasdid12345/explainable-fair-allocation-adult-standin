"""Baseline (unconstrained logistic regression) vs improved (class-balanced LightGBM
with per-sex thresholds tuned on validation data) on OpenML Adult (data_id 1590).
Sex is the sensitive attribute; predicted positives are treated as 'allocations'."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # both models are run with these seeds
MAX_ROWS = 200_000            # subsample cap to keep runtime small
DATA_ID = 1590
SOURCE_URL = "https://www.openml.org/d/1590"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_adult(max_tries=5):
    """Download Adult via openml; retry with exponential backoff on transient errors."""
    for attempt in range(max_tries):
        try:
            ds = openml.datasets.get_dataset(DATA_ID)
            X, y, _, _ = ds.get_data(dataset_format="dataframe",
                                     target=ds.default_target_attribute)
            return X, y
        except Exception as exc:  # network error, HTTP 429 or 5xx
            wait = 2 ** attempt
            log(f"download failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("could not download OpenML dataset 1590")


def evaluate(pred, y, sex):
    """Shared evaluation for both models: demographic parity gap, equal-opportunity gap,
    accuracy and F1 on a given split. pred/y are 0/1 arrays; sex is 'Female'/'Male'."""
    f = sex == "Female"
    dp = abs(pred[f].mean() - pred[~f].mean())
    tpr_f = pred[f & (y == 1)].mean()
    tpr_m = pred[~f & (y == 1)].mean()
    return dict(dp=dp, eo=abs(tpr_f - tpr_m),
                acc=accuracy_score(y, pred), f1=f1_score(y, pred))


def pick_group_thresholds(p, y, sex, grid=np.arange(0.05, 0.96, 0.01)):
    """Validation-only search: one threshold per sex, minimising the allocation-rate gap;
    ties broken by higher validation accuracy."""
    f = sex == "Female"
    best_key, best = None, None
    for tf in grid:
        for tm in grid:
            pred = np.where(f, p >= tf, p >= tm)
            gap = abs(pred[f].mean() - pred[~f].mean())
            key = (round(gap, 4), -(pred == y).mean())
            if best_key is None or key < best_key:
                best_key, best = key, (tf, tm)
    return best


def main():
    X, y = fetch_adult()
    X = X.replace("?", np.nan).reset_index(drop=True)
    y = y.astype(str).str.contains(">50K").astype(int).to_numpy()
    if len(X) > MAX_ROWS:  # subsample very large data
        keep = np.random.RandomState(SEED).choice(len(X), MAX_ROWS, replace=False)
        X, y = X.iloc[keep].reset_index(drop=True), y[keep]
    cat_cols = list(X.select_dtypes(["category", "object"]).columns)
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:  # category dtype for LightGBM native handling
        X[c] = X[c].astype("category")
    X_lr = X.copy()  # LogisticRegression pipeline needs plain object dtype
    X_lr[cat_cols] = X[cat_cols].astype(object)
    sex = X["sex"].astype(str).to_numpy()
    n = len(X)
    log(f"loaded {n} rows, {len(cat_cols)} categorical and {len(num_cols)} numeric features")

    # Split ONCE: 75/25 stratified test split; training part split 80/20 into fit/validation.
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.25, stratify=y, random_state=SEED)
    idx_fit, idx_val = train_test_split(idx_tr, test_size=0.2, stratify=y[idx_tr], random_state=SEED)
    log(f"train={len(idx_tr)} fit={len(idx_fit)} val={len(idx_val)} test={len(idx_te)}")

    base_runs, imp_runs, thresholds = [], [], []
    for seed in SEEDS:
        # BASELINE: unconstrained logistic regression, single 0.5 threshold.
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), cat_cols)])
        lr = make_pipeline(pre, LogisticRegression(max_iter=1000, random_state=seed))
        lr.fit(X_lr.iloc[idx_fit], y[idx_fit])
        p_te = lr.predict_proba(X_lr.iloc[idx_te])[:, 1]
        base_runs.append(evaluate((p_te >= 0.5).astype(int), y[idx_te], sex[idx_te]))

        # IMPROVED: class-balanced LightGBM; per-sex thresholds chosen on validation split.
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 class_weight="balanced", random_state=seed, verbose=-1)
        gbm.fit(X.iloc[idx_fit], y[idx_fit])
        tf, tm = pick_group_thresholds(gbm.predict_proba(X.iloc[idx_val])[:, 1],
                                       y[idx_val], sex[idx_val])
        p_te = gbm.predict_proba(X.iloc[idx_te])[:, 1]
        pred = np.where(sex[idx_te] == "Female", p_te >= tf, p_te >= tm).astype(int)
        imp_runs.append(evaluate(pred, y[idx_te], sex[idx_te]))
        thresholds.append((float(tf), float(tm)))
        log(f"seed {seed}: baseline DP={base_runs[-1]['dp']:.4f} "
            f"improved DP={imp_runs[-1]['dp']:.4f} thresholds F={tf:.2f} M={tm:.2f}")

    def agg(runs, key):
        vals = np.array([r[key] for r in runs], dtype=float)
        return float(vals.mean()), float(vals.std())

    b_dp, b_dp_sd = agg(base_runs, "dp")
    i_dp, i_dp_sd = agg(imp_runs, "dp")
    b_acc, b_acc_sd = agg(base_runs, "acc")
    i_acc, i_acc_sd = agg(imp_runs, "acc")
    b_eo, _ = agg(base_runs, "eo")
    i_eo, _ = agg(imp_runs, "eo")
    b_f1, _ = agg(base_runs, "f1")
    i_f1, _ = agg(imp_runs, "f1")
    values = [b_dp, i_dp, b_acc, i_acc, b_eo, i_eo, b_f1, i_f1]
    if not all(np.isfinite(values)) or len(idx_te) == 0:
        log("metrics not meaningful; exiting")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    names = ["Baseline LR", "Improved LGBM"]
    ax[0].bar(names, [b_dp, i_dp]); ax[0].set_title("Demographic parity gap (lower better)")
    ax[1].bar(names, [b_acc, i_acc]); ax[1].set_title("Test accuracy (higher better)")
    fig.tight_layout(); fig.savefig("figures/comparison.png", dpi=120); plt.close(fig)

    results = {
        "question": ("Can a class-balanced LightGBM with validation-tuned, group-specific decision "
                     "thresholds give a smaller sex-based demographic parity gap in allocation "
                     "decisions than a standard unconstrained logistic regression on the public "
                     "Adult dataset, at a small cost in accuracy?"),
        "method": ("Stand-in for multi-agent reward allocation: each person is an agent, income >50K "
                   "is the reward-worthy outcome, sex is the sensitive attribute. Stratified 75/25 "
                   "train/test split (seed 42); training part split 80/20 into fit/validation. "
                   "Baseline: unconstrained logistic regression, single 0.5 threshold. Improved: "
                   "class-balanced LightGBM with per-sex thresholds chosen on validation to equalise "
                   "allocation rates. Models run with seeds 42, 43, 44."),
        "dataset": {
            "name": "Adult (OpenML data_id 1590)",
            "source_url": SOURCE_URL,
            "n_rows": int(n),
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "demographic parity difference on test (|allocation rate F - M|)",
        "higher_is_better": False,
        "baseline": "Unconstrained standard logistic regression (one-hot categoricals, standardised numerics), predicted positives at threshold 0.5.",
        "improved": "Class-balanced LightGBM (fit on fit split) with separate thresholds for Female and Male chosen on the validation split to minimise the allocation-rate gap (ties broken by validation accuracy).",
        "metrics": {
            "baseline_score": b_dp,
            "improved_score": i_dp,
            "improvement": b_dp - i_dp,
            "baseline_score_std": b_dp_sd,
            "improved_score_std": i_dp_sd,
            "baseline_accuracy": b_acc,
            "improved_accuracy": i_acc,
            "baseline_accuracy_std": b_acc_sd,
            "improved_accuracy_std": i_acc_sd,
            "baseline_eo_gap": b_eo,
            "improved_eo_gap": i_eo,
            "baseline_f1": b_f1,
            "improved_f1": i_f1,
        },
        "notes": ("Single stratified split (seed 42); seeds 42-44 vary only model randomness. "
                  "No hyperparameter search was run; LightGBM settings are fixed. Thresholds are "
                  "selected on the validation split, never on test. Both models are trained on the "
                  "fit split only. Sex is included as a feature in both models. Standard deviations "
                  "are population std (ddof=0) over 3 seeds; logistic regression is deterministic so "
                  "its std is 0. Demographic parity ignores base-rate differences and the "
                  "equal-opportunity gap is reported as a secondary fairness check. Results are "
                  "from the public Adult data, not the paper's reinforcement-learning setting."),
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log(f"baseline DP={b_dp:.4f} improved DP={i_dp:.4f}; wrote results.json")


if __name__ == "__main__":
    main()
