"""Fit the final cancellation-risk model on the TRAINING table only and save it next to predict.py.

Run from the project root:  ./.venv/bin/python deliverable/train.py  (paths are relative to this file)
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict as P  # shared cleaning / encoding so train and score are identical

DB = HERE.parent / "train_val.db"
SEED = 42
NUM_BOOST_ROUND = 600
PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    cat_smooth=20, min_data_per_group=100, max_cat_to_onehot=4,
    seed=SEED, deterministic=True, force_row_wise=True, verbose=-1, num_threads=4,
)


def target(df):
    return (df["is_canceled"].astype(str).str.strip().str.lower()
            .map({"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}).astype(int))


def main():
    t0 = time.time()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    raw = pd.read_sql_query("SELECT * FROM train", con)
    con.close()

    raw = raw.drop_duplicates()  # 244 exact duplicate rows
    y = target(raw).values
    clean = P.clean(raw)
    raw_adr = raw["adr"].map(P._parse_money)
    keep = (~(raw_adr > P.ADR_CAP)).values  # drop rows with corrupt (uncapped) ADR; NaN kept
    clean, y = clean[keep].reset_index(drop=True), y[keep]

    vocab = P.fit_vocab(clean, min_count=50)
    X = P.encode(clean, vocab)
    cats = list(range(len(P.NUM_FEATURES), len(P.NUM_FEATURES) + len(P.CAT_FEATURES)))
    ds = lgb.Dataset(X.values, y, feature_name=list(X.columns), categorical_feature=cats)
    booster = lgb.train(PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)

    booster.save_model(str(HERE / "model.txt"))
    meta = {
        "features": list(X.columns),
        "num_features": P.NUM_FEATURES,
        "cat_features": P.CAT_FEATURES,
        "vocab": vocab,
        "adr_cap": P.ADR_CAP,
        "num_boost_round": NUM_BOOST_ROUND,
        "params": PARAMS,
        "train_rows_used": int(len(y)),
        "train_pos_rate": float(y.mean()),
    }
    (HERE / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"trained on {len(y)} rows, pos rate {y.mean():.4f}, in {time.time() - t0:.1f}s")
    print("saved", HERE / "model.txt", HERE / "model_meta.json")


if __name__ == "__main__":
    main()
