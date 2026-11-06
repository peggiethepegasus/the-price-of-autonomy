"""Hotel booking cancellation-risk scorer.

predict_proba(db_path) reads the table `features` from the SQLite file at db_path
(raw, messy booking rows), cleans it, scores every row with the saved LightGBM model
(model.txt + model_meta.json in this folder) and returns booking_id / risk_score.

Only information that exists at the moment a booking is created is used. Columns
that are only known later (reservation_status, reservation_status_date,
booking_changes, total_of_special_requests, assigned_room_type, is_canceled) are
never read.
"""

import json
import re
import sqlite3
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.txt"
META_PATH = HERE / "model_meta.json"

TIME_ORIGIN = pd.Timestamp("2023-01-01")
# ADR above this is treated as corrupt: 1,225 City Hotel rows (Mar 2024) carry ADR 1,900-7,100
# while the hotel's 95th percentile is ~450. Training drops them; scoring caps at this value.
ADR_CAP = 1000.0
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MISSING_TOKENS = {"", "-", "na", "n/a", "nan", "null", "none", "unknown", "undefined"}
COUNTRY_NAMES = {
    "australia": "aus", "china": "chn", "hong kong": "hkg", "india": "ind", "indonesia": "idn",
    "japan": "jpn", "malaysia": "mys", "myanmar": "mmr", "new zealand": "nzl", "other": "oth",
    "philippines": "phl", "singapore": "sgp", "south korea": "kor", "taiwan": "twn",
    "thailand": "tha", "united kingdom": "gbr", "united states": "usa", "vietnam": "vnm",
}
DEPOSIT_FIX = {"no depost": "no deposit"}
SEGMENT_FIX = {"onlin ta": "online ta", "offline ta/t0": "offline ta/to"}
CUSTOMER_FIX = {"transiant": "transient"}

NUM_FEATURES = [
    "lead_time", "arrival_year", "arrival_month", "arrival_day", "arrival_dow",
    "arrival_week", "arrival_doy", "arrival_days", "stays_in_weekend_nights",
    "stays_in_week_nights", "total_nights", "adults", "children", "babies",
    "total_guests", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "adr",
]
CAT_FEATURES = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "reserved_room_type", "agent", "promo_code",
]
RAW_COLUMNS = ["booking_id"] + [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "deposit_type", "agent", "customer_type", "adr", "promo_code",
]


def _text(v):
    """Lower-case, collapse whitespace, map placeholder tokens to 'unknown'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = re.sub(r"\s+", " ", str(v).strip()).lower()
    return "unknown" if s in MISSING_TOKENS else s


def _num(s):
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.str.lower().isin(["", "none", "nan", "null"])), errors="coerce")


def _parse_money(v):
    """'172.50', '172,50', 'SGD 172.50', '1.234,50', '0,00' -> float; junk -> NaN."""
    if v is None:
        return np.nan
    s = re.sub(r"[A-Za-z$€£\s]", "", str(v).strip())
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw messy text table -> typed, normalised feature frame (no encoding yet)."""
    out = pd.DataFrame(index=raw.index)
    out["booking_id"] = raw["booking_id"].values

    # --- dates (arrival); week number is re-derived from the date because the raw one is noisy
    year = _num(raw["arrival_date_year"])
    mon_txt = raw["arrival_date_month"].astype(str).str.strip().str.lower().str[:3]
    month = mon_txt.map(MONTHS)
    month = month.fillna(pd.to_numeric(raw["arrival_date_month"], errors="coerce"))
    day = _num(raw["arrival_date_day_of_month"])
    date = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce")
    out["arrival_year"] = year
    out["arrival_month"] = month
    out["arrival_day"] = day
    out["arrival_dow"] = date.dt.dayofweek
    out["arrival_week"] = date.dt.isocalendar().week.astype("float")
    out["arrival_doy"] = date.dt.dayofyear
    out["arrival_days"] = (date - TIME_ORIGIN).dt.days

    # --- lead time: '14 days' style strings; negative lead times are entry errors -> 0
    lt = raw["lead_time"].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    out["lead_time"] = pd.to_numeric(lt, errors="coerce").clip(lower=0)

    # --- stay and party size
    out["stays_in_weekend_nights"] = _num(raw["stays_in_weekend_nights"])
    out["stays_in_week_nights"] = _num(raw["stays_in_week_nights"])
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    for c in ["adults", "children", "babies"]:
        out[c] = _num(raw[c])
    out["total_guests"] = out["adults"] + out["children"].fillna(0) + out["babies"].fillna(0)

    # --- customer history
    out["is_repeated_guest"] = _num(raw["is_repeated_guest"])
    out["previous_cancellations"] = _num(raw["previous_cancellations"])
    out["previous_bookings_not_canceled"] = _num(raw["previous_bookings_not_canceled"])

    # --- price
    out["adr"] = raw["adr"].map(_parse_money).astype(float).clip(upper=ADR_CAP)

    # --- categoricals: normalise spelling / case / placeholders
    out["hotel"] = raw["hotel"].map(_text)
    meal = raw["meal"].map(_text)
    out["meal"] = meal
    out["country"] = raw["country"].map(lambda v: COUNTRY_NAMES.get(_text(v), _text(v)))
    seg = raw["market_segment"].map(lambda v: _text(v).replace("-", " "))
    seg = seg.map(lambda s: SEGMENT_FIX.get(re.sub(r"\s+", " ", s), re.sub(r"\s+", " ", s)))
    out["market_segment"] = seg
    out["distribution_channel"] = raw["distribution_channel"].map(_text)
    dep = raw["deposit_type"].map(lambda v: _text(v).replace("-", " "))
    dep = dep.map(lambda s: DEPOSIT_FIX.get(re.sub(r"\s+", " ", s), re.sub(r"\s+", " ", s)))
    out["deposit_type"] = dep
    cust = raw["customer_type"].map(_text)
    out["customer_type"] = cust.map(lambda s: CUSTOMER_FIX.get(s, s))
    out["reserved_room_type"] = raw["reserved_room_type"].map(_text)
    ag = raw["agent"].astype(str).str.strip()
    ag_num = pd.to_numeric(ag, errors="coerce")
    out["agent"] = np.where(ag_num.notna(), ag_num.fillna(-1).astype(int).astype(str), "none")
    promo = raw["promo_code"].astype(str).str.strip().str.upper()
    out["promo_code"] = promo.where(~promo.isin(["", "NAN", "NONE", "NULL"]), "NONE")
    return out


def fit_vocab(clean_df: pd.DataFrame, min_count: int = 50) -> dict:
    """Keep categories seen at least min_count times in the training table; rest -> OTHER."""
    vocab = {}
    for c in CAT_FEATURES:
        vc = clean_df[c].value_counts()
        keep = sorted(vc[vc >= min_count].index.astype(str).tolist())
        vocab[c] = keep
    return vocab


def encode(clean_df: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Numeric features + integer codes for categoricals (unseen -> OTHER code)."""
    X = clean_df[NUM_FEATURES].astype(float).copy()
    for c in CAT_FEATURES:
        cats = vocab[c]
        other = len(cats)
        lookup = {v: i for i, v in enumerate(cats)}
        X[c] = clean_df[c].astype(str).map(lambda v: lookup.get(v, other)).astype(float)
    return X[NUM_FEATURES + CAT_FEATURES]


_MODEL = None
_META = None


def _load():
    global _MODEL, _META
    if _MODEL is None:
        _META = json.loads(META_PATH.read_text())
        _MODEL = lgb.Booster(model_file=str(MODEL_PATH))
    return _MODEL, _META


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    model, meta = _load()
    clean_df = clean(raw)
    X = encode(clean_df, meta["vocab"])
    cols = meta["features"]
    p = model.predict(X[cols].to_numpy(dtype=float))
    return pd.DataFrame({
        "booking_id": raw["booking_id"].astype(str).values,
        "risk_score": np.clip(np.asarray(p, dtype=float), 0.0, 1.0),
    })


def predict_proba(db_path):
    """Score every row of the `features` table in db_path. Returns booking_id, risk_score."""
    uri = Path(db_path).expanduser().resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        raw = pd.read_sql_query("SELECT * FROM features", con)
    finally:
        con.close()
    return score_frame(raw)
