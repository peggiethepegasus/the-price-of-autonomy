"""Cleaning and feature construction for the benchmark model.

The raw hotel table stores every column as messy text (mixed spellings, 'SGD' prefixes, 'days' suffixes, typos such as 'No Depost').
clean() turns a raw frame into the 24 model inputs that are all known at the moment a booking is created; to_model_frame() then
gives the 10 categorical columns a pandas 'category' dtype with levels taken from the TRAINING table only.
Used both to train the benchmark (notebook) and inside the benchmark's predict.py when the evaluator scores it.
"""
import re
import numpy as np
import pandas as pd

NUMERIC = ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
           "previous_cancellations", "previous_bookings_not_canceled", "total_of_special_requests"]
CATEGORICAL = ["hotel", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type",
               "deposit_type", "agent", "customer_type", "promo_code"]
FEATURES = (["lead_time", "lead_time_was_negative", "lead_time_was_capped", "adr", "is_corporate_booking"] + NUMERIC + CATEGORICAL)
MISSING = {"", "-", "UNKNOWN", "NULL", "N/A", "NA", "NAN"}
ISO3 = {"SGP", "MYS", "IDN", "CHN", "IND", "AUS", "USA", "GBR", "PHL", "JPN", "KOR", "THA", "VNM", "HKG", "TWN", "NZL", "MMR", "OTH"}
COUNTRY_NAMES = {"SINGAPORE": "SGP", "MALAYSIA": "MYS", "INDONESIA": "IDN", "CHINA": "CHN", "INDIA": "IND", "AUSTRALIA": "AUS",
                 "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA", "UNITED KINGDOM": "GBR", "PHILIPPINES": "PHL", "JAPAN": "JPN",
                 "SOUTH KOREA": "KOR", "THAILAND": "THA", "VIETNAM": "VNM", "HONG KONG": "HKG", "TAIWAN": "TWN", "NEW ZEALAND": "NZL",
                 "MYANMAR": "MMR", "OTHER": "OTH"}


def _tok(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else re.sub(r"\s+", " ", str(v).strip()).upper()


def _squash(t):
    return re.sub(r"[^A-Z0-9]+", "", t)


def _deposit(v):
    t = re.sub(r"\s+", " ", str(v).strip()).lower()
    if re.fullmatch(r"no[\s-]*depo(?:sit|st)", t): return "No Deposit"
    if re.fullmatch(r"non[\s-]*refund", t): return "Non Refund"
    if re.fullmatch(r"refundable", t): return "Refundable"
    return str(v).strip()


def _meal(v):
    t = _tok(v)
    if t == "": return "Undefined"
    t = re.sub(r"[\s/_-]+", " ", t); t = re.sub(r"[^A-Z& ]", "", t).strip()
    if re.fullmatch(r"BB|B B|BED ?& ?BREAKFAST|BED AND BREAKFAST", t): return "BB"
    if re.fullmatch(r"HB|H B|HALF BOARD", t): return "HB"
    if re.fullmatch(r"FB|F B|FULL BOARD", t): return "FB"
    if re.fullmatch(r"SC|S C|SELF CATERING|SELF CATERING ONLY", t): return "SC"
    return "Undefined"


def _country(v):
    t = _tok(v)
    if t in MISSING or t in {"OTH", "OTHER"}: return "OTH"
    if t in ISO3: return t
    return COUNTRY_NAMES.get(t, "OTH")


def _norm_channel(v):
    t = _tok(v)
    if t in MISSING: return None
    t = t.replace("T0", "TO").replace("-", " ")
    t = re.sub(r"\s*/\s*", "/", t); t = re.sub(r"\s+", " ", t).strip()
    return _squash(t).replace("0", "O")


def _market(v):
    k = _norm_channel(v)
    return {"ONLINETA": "Online TA", "ONLINTA": "Online TA", "OFFLINETATO": "Offline TA/TO", "GROUPS": "Groups", "DIRECT": "Direct",
            "CORPORATE": "Corporate", "COMPLEMENTARY": "Complementary", "AVIATION": "Aviation"}.get(k, "Undefined")


def _channel(v):
    k = _norm_channel(v)
    return {"TATO": "TA/TO", "DIRECT": "Direct", "CORPORATE": "Corporate", "GDS": "GDS"}.get(k, "Undefined")


def _customer(v):
    t = _tok(v)
    if t in MISSING: return "Unknown"
    k = _squash(t.replace("-", " "))
    if k == "TRANSIANT": k = "TRANSIENT"
    return {"TRANSIENT": "Transient", "TRANSIENTPARTY": "Transient-Party", "CONTRACT": "Contract", "GROUP": "Group"}.get(k, "Unknown")


def arrival_date(df):
    mk = df["arrival_date_month"].astype(str).str.strip().str.replace(r"[^A-Za-z]", "", regex=True).str[:3].str.lower()
    mm = mk.map({m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())})
    return pd.to_datetime({"year": pd.to_numeric(df["arrival_date_year"], errors="coerce"), "month": mm,
                           "day": pd.to_numeric(df["arrival_date_day_of_month"], errors="coerce")}, errors="coerce")


def clean(df):
    """Raw 'features'-style frame -> model frame (FEATURES columns, categoricals still plain strings)."""
    o = pd.DataFrame(index=df.index)
    lt = pd.to_numeric(df["lead_time"].astype(str).str.strip().str.replace(r"(?i)\s*days?\s*", "", regex=True), errors="coerce").fillna(0)
    o["lead_time_was_negative"] = (lt < 0).astype(int); o["lead_time_was_capped"] = (lt > 730).astype(int)
    o["lead_time"] = lt.clip(lower=0, upper=730)
    adr = df["adr"].astype(str).str.strip().str.replace(r"(?i)^SGD\s*", "", regex=True).str.replace(",", ".", regex=False)
    o["adr"] = pd.to_numeric(adr, errors="coerce").fillna(0.0)
    comp = df["company"].astype("string").str.strip()
    o["is_corporate_booking"] = (~(comp.isna() | comp.eq(""))).astype(int)
    for c in NUMERIC:
        o[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    o["hotel"] = df["hotel"].astype(str).str.strip()
    o["meal"] = df["meal"].map(_meal)
    o["country"] = df["country"].map(_country)
    o["market_segment"] = df["market_segment"].map(_market)
    o["distribution_channel"] = df["distribution_channel"].map(_channel)
    o["reserved_room_type"] = df["reserved_room_type"].astype(str).str.strip()
    o["deposit_type"] = df["deposit_type"].map(_deposit)
    ag = df["agent"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True).fillna("")
    o["agent"] = ag.mask(ag.eq(""), "No Agent").astype(object)
    o["customer_type"] = df["customer_type"].map(_customer)
    pr = df["promo_code"].astype("string").str.strip().fillna("")
    o["promo_code"] = pr.mask(pr.eq(""), "None").astype(object)
    return o[FEATURES]


def to_model_frame(cleaned, levels):
    """Categoricals -> pandas category with TRAIN-ONLY levels (unseen -> missing)."""
    x = cleaned.copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c].astype(str), categories=levels[c])
    return x
