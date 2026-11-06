from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import joblib

_MODEL = joblib.load(Path(__file__).with_name('model.joblib'))
NUM = _MODEL['num']
CAT = _MODEL['cat']
COLUMNS = _MODEL['columns']
CAT_LEVELS = _MODEL['cat_levels']


def _clean(df):
    x = pd.DataFrame(index=df.index)
    for c in NUM:
        s = df[c].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'NULL': np.nan, 'None': np.nan})
        x[c] = pd.to_numeric(s, errors='coerce')
    months = {'Jan': 1, 'January': 1, 'Feb': 2, 'February': 2, 'Mar': 3, 'March': 3,
              'Apr': 4, 'April': 4, 'May': 5, 'Jun': 6, 'June': 6, 'Jul': 7, 'July': 7,
              'Aug': 8, 'August': 8, 'Sep': 9, 'September': 9, 'Oct': 10, 'October': 10,
              'Nov': 11, 'November': 11, 'Dec': 12, 'December': 12}
    x['arrival_month_num'] = df['arrival_date_month'].astype(str).str.strip().map(months).fillna(0)
    x['total_nights'] = x['stays_in_weekend_nights'].fillna(0) + x['stays_in_week_nights'].fillna(0)
    x['party_size'] = x['adults'].fillna(0) + x['children'].fillna(0) + x['babies'].fillna(0)
    x['room_changed'] = (df['reserved_room_type'].astype(str).str.strip() != df['assigned_room_type'].astype(str).str.strip()).astype('int8')
    for c in CAT:
        s = df[c].astype(str).str.strip().replace({'': '__MISSING__', 'nan': '__MISSING__', 'None': '__MISSING__'})
        x[c] = pd.Categorical(s, categories=CAT_LEVELS[c]).codes.astype('int32')
    return x[COLUMNS]


def predict_proba(db_path):
    """Return booking_id and calibrated at-risk probability for every features row."""
    with sqlite3.connect(db_path) as con:
        raw = pd.read_sql_query('SELECT * FROM features', con)
    features = _clean(raw)
    scores = _MODEL['model'].predict_proba(features)[:, 1]
    return pd.DataFrame({'booking_id': raw['booking_id'].to_numpy(), 'risk_score': np.clip(scores, 0.0, 1.0)})
