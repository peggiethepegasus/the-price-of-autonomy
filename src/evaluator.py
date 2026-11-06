"""Independent evaluator: scores any deliverable (a folder with predict.py) the same way for every entrant.

Protocol (identical for the benchmark, the floors and every LLM agent):
  * predict.py must expose predict_proba(db_path) -> DataFrame[booking_id, risk_score]; it reads the table `features` of a SQLite file.
  * VALIDATION: the full validation table exactly as the agents saw it (post-outcome columns included).
  * TEST: data/test.db `features` (the five post-outcome columns are NULL). A model that depends on them collapses here.
  * LEAK TEST: the deliverable is scored a second time on validation with the five post-outcome columns blanked (NULL).
    If PR-AUC changes by more than 0.0005 (or scoring fails) the model is flagged as relying on leaky columns.
    The test is then repeated one column at a time to show WHICH column(s) it relied on.
  * Each booking_id is scored once (first occurrence). Headline metric: PR-AUC of the at-risk class (positive = cancelled or no-show).
  * Threshold = F1-maximising point on validation, applied unchanged to test (or a fixed one for trivial floors).
  * 95% intervals come from a bootstrap of the test bookings (descriptive; not a significance test).

Usage:  from evaluator import evaluate;  res = evaluate("path/to/deliverable", name="my_model")
"""
import contextlib, json, shutil, sqlite3, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_VAL_DB, TEST_DB = ROOT / "data" / "train_val.db", ROOT / "data" / "test.db"
POST_OUTCOME = ["reservation_status", "reservation_status_date", "is_canceled", "assigned_room_type", "booking_changes"]
LEAK_DROP_THRESHOLD = 0.0005
LOADER = """import importlib.util, sys
d, db, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("agent_predict", d + "/predict.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.predict_proba(db).to_csv(out, index=False)
"""


def _read(db, sql):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try: return pd.read_sql(sql, con)
    finally: con.close()


def _write_features(df, path):
    path.unlink(missing_ok=True); c = sqlite3.connect(path); df.to_sql("features", c, index=False); c.close()


def _run_predict(python_exe, deliverable, db_path, out_csv, timeout=1200):
    t = time.time()
    p = subprocess.run([python_exe, "-c", LOADER, str(deliverable), str(db_path), str(out_csv)], cwd=str(deliverable),
                       capture_output=True, text=True, timeout=timeout)
    return time.time() - t, p.returncode, (p.stderr or "")[-1500:]


def _load_scores(csv, ids):
    df = pd.read_csv(csv, dtype={"booking_id": str})
    if not {"booking_id", "risk_score"} <= set(df.columns): raise ValueError(f"need columns booking_id,risk_score; got {list(df.columns)}")
    s = pd.to_numeric(df.drop_duplicates("booking_id").set_index("booking_id")["risk_score"], errors="coerce")
    if set(ids) - set(s.index): raise ValueError("some booking_ids are missing from the predictions")
    s = s.reindex(ids)
    if s.isna().any(): raise ValueError("NaN / non-numeric scores")
    return s.values


def _best_f1_threshold(y, s):
    p, r, thr = precision_recall_curve(y, s)
    f = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12), 0)
    return float(thr[int(np.argmax(f))])


def _point_metrics(y, s, thr):
    yhat = (s >= thr).astype(int)
    return dict(accuracy=float(accuracy_score(y, yhat)), precision=float(precision_score(y, yhat, zero_division=0)),
                recall=float(recall_score(y, yhat, zero_division=0)), f1=float(f1_score(y, yhat, zero_division=0)), flagged_share=float(yhat.mean()))


def _bootstrap_ci(y, s, n=1000, seed=42):
    rng = np.random.default_rng(seed); ap = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), replace=True)
        if y[b].min() != y[b].max(): ap.append(average_precision_score(y[b], s[b]))
    return [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))]


def evaluate(deliverable, name="model", python_exe=None, fixed_threshold=None, per_column=True, bootstrap=True):
    """Score one deliverable folder and return a dict of results. Nothing is written into the deliverable folder."""
    deliverable, python_exe = Path(deliverable).resolve(), python_exe or sys.executable
    res = {"name": name, "status": "ok", "errors": {}}
    tmp = Path(tempfile.mkdtemp(prefix="eval_"))
    try:
        if not (deliverable / "predict.py").exists(): return {**res, "status": "no_predict_py"}
        val = _read(TRAIN_VAL_DB, "SELECT * FROM validation")
        v_ids = list(dict.fromkeys(val.booking_id))
        yv = (val.drop_duplicates("booking_id").set_index("booking_id").reindex(v_ids).reservation_status != "Check-Out").astype(int).values
        # --- validation
        _write_features(val, tmp / "val.db"); sec, rc, err = _run_predict(python_exe, deliverable, tmp / "val.db", tmp / "val.csv")
        res["validation"] = {"predict_seconds": round(sec, 1)}
        try:
            if rc: raise RuntimeError("predict.py failed on validation:\n" + err)
            sv = _load_scores(tmp / "val.csv", v_ids)
        except Exception as e:
            res["status"], res["errors"]["validation"], sv = "validation_scoring_failed", str(e)[-1500:], None
        # --- leak test (all five post-outcome columns blanked, then one at a time)
        lt = {"columns_blanked": POST_OUTCOME}
        if sv is None: lt.update(status="not_run", leaky=None)
        else:
            _write_features(val.assign(**{c: None for c in POST_OUTCOME}), tmp / "blank.db")
            _, rc2, err2 = _run_predict(python_exe, deliverable, tmp / "blank.db", tmp / "blank.csv")
            try:
                if rc2: raise RuntimeError(err2)
                full, stripped = float(average_precision_score(yv, sv)), float(average_precision_score(yv, _load_scores(tmp / "blank.csv", v_ids)))
                lt.update(status="ok", pr_auc_full=full, pr_auc_blanked=stripped, drop=full - stripped, leaky=bool(abs(full - stripped) > LEAK_DROP_THRESHOLD))
            except Exception as e:
                lt.update(status="scoring_failed", leaky=True, error=str(e)[-600:])
            if lt.get("status") == "ok" and per_column and lt["leaky"]:
                per = {}
                for col in POST_OUTCOME:
                    _write_features(val.assign(**{col: None}), tmp / "c.db")
                    _, rc3, e3 = _run_predict(python_exe, deliverable, tmp / "c.db", tmp / "c.csv")
                    try:
                        if rc3: raise RuntimeError(e3)
                        per[col] = full - float(average_precision_score(yv, _load_scores(tmp / "c.csv", v_ids)))
                    except Exception: per[col] = None            # scoring failed => the model needs this column
                lt["per_column_drop"] = per
                lt["columns_relied_on"] = [k for k, v in per.items() if v is None or abs(v) > LEAK_DROP_THRESHOLD]
        res["leak_test"] = lt
        # --- test (labels live in the same file; only the features table is shown to predict.py)
        feats, labels = _read(TEST_DB, "SELECT * FROM features"), _read(TEST_DB, "SELECT * FROM labels")
        t_ids, yt = list(labels.booking_id), labels.at_risk.values.astype(int)
        _write_features(feats, tmp / "test.db"); sec, rc, err = _run_predict(python_exe, deliverable, tmp / "test.db", tmp / "test.csv")
        res["test"] = {"predict_seconds": round(sec, 1)}
        try:
            if rc: raise RuntimeError("predict.py failed on test features:\n" + err)
            st = _load_scores(tmp / "test.csv", t_ids)
        except Exception as e:
            if res["status"] == "ok": res["status"] = "test_scoring_failed"
            res["errors"]["test"], st = str(e)[-1500:], None
        thr = fixed_threshold
        if sv is not None:
            res["validation"].update(pr_auc=float(average_precision_score(yv, sv)), roc_auc=float(roc_auc_score(yv, sv)), prevalence=float(yv.mean()))
            if thr is None: thr = _best_f1_threshold(yv, sv)
        res["threshold"] = thr
        if st is not None:
            res["test"].update(pr_auc=float(average_precision_score(yt, st)), roc_auc=float(roc_auc_score(yt, st)), prevalence=float(yt.mean()), scores=st, labels=yt)
            if thr is not None: res["test"]["at_threshold"] = _point_metrics(yt, st, thr)
            if bootstrap: res["test"]["pr_auc_95ci"] = _bootstrap_ci(yt, st)
            if sv is not None: res["gap_val_minus_test_pr_auc"] = res["validation"]["pr_auc"] - res["test"]["pr_auc"]
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def summary_row(res):
    t, v, lt = res.get("test", {}), res.get("validation", {}), res.get("leak_test", {})
    ci = t.get("pr_auc_95ci")
    return {"name": res["name"], "status": res["status"], "val_PR_AUC": v.get("pr_auc"), "test_PR_AUC": t.get("pr_auc"),
            "test_95%_interval": f"{ci[0]:.4f}-{ci[1]:.4f}" if ci else "", "leaky": lt.get("leaky"), "relied_on": "|".join(lt.get("columns_relied_on", []))}
