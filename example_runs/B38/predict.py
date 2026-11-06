"""Score raw hotel bookings for cancellation/no-show risk."""

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

NUMERIC_RAW = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'adr', 'total_of_special_requests',
]
CATEGORICAL_RAW = [
    'hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'deposit_type', 'agent', 'company', 'customer_type', 'promo_code',
]
MONTHS = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
}
COUNTRY_MAP = {
    'singapore': 'SGP', 'malaysia': 'MYS', 'indonesia': 'IDN', 'china': 'CHN',
    'india': 'IND', 'australia': 'AUS', 'united states': 'USA', 'united kingdom': 'GBR',
    'japan': 'JPN', 'south korea': 'KOR', 'hong kong': 'HKG', 'myanmar': 'MMR',
    'new zealand': 'NZL', 'philippines': 'PHL', 'taiwan': 'TWN', 'thailand': 'THA',
    'vietnam': 'VNM', 'other': 'OTH',
}
MISSING = {'', '-', 'null', 'none', 'nan', 'na', 'n/a', 'unknown'}


def _normalize_category(series, column):
    values = series.astype('string').fillna('').str.strip().str.replace(r'\s+', ' ', regex=True)
    values = values.str.lower().mask(values.str.lower().isin(MISSING), '__missing__')
    if column == 'country':
        values = values.replace({key: value.lower() for key, value in COUNTRY_MAP.items()})
    elif column == 'meal':
        values = values.replace({'undefined': '__missing__'})
    elif column == 'market_segment':
        values = values.str.replace('-', ' ', regex=False).str.replace(r'\s+', ' ', regex=True)
        values = values.replace({
            'onlin ta': 'online ta', 'offline ta/t0': 'offline ta/to',
            'undefined': '__missing__',
        })
    elif column == 'distribution_channel':
        values = values.replace({'undefined': '__missing__'})
    elif column == 'deposit_type':
        values = values.str.replace('-', ' ', regex=False).str.replace(r'\s+', ' ', regex=True)
        values = values.replace({'no depost': 'no deposit'})
    elif column == 'customer_type':
        values = values.str.replace(r'\s*-\s*', '-', regex=True)
        values = values.replace({'transiant': 'transient'})
    elif column == 'agent':
        numbers = pd.to_numeric(values.mask(values.eq('__missing__')), errors='coerce')
        values = numbers.map(
            lambda value: str(int(value))
            if pd.notna(value) and np.isfinite(value) and float(value).is_integer()
            else (str(value) if pd.notna(value) and np.isfinite(value) else '__missing__')
        ).astype('string')
    return values.astype(str)


def _clean_features(raw):
    cleaned = pd.DataFrame(index=raw.index)
    for column in NUMERIC_RAW:
        source = raw[column] if column in raw else pd.Series(np.nan, index=raw.index)
        cleaned[column] = pd.to_numeric(
            source.astype('string').str.strip(), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan).astype(float)

    cleaned['lead_time_invalid'] = (
        (cleaned['lead_time'] < 0) | (cleaned['lead_time'] > 1000)
    ).astype(float)
    cleaned['lead_time'] = cleaned['lead_time'].clip(lower=0, upper=1000)
    if 'arrival_date_month' in raw:
        cleaned['arrival_month'] = (
            raw['arrival_date_month'].astype('string').str.strip().str.lower().map(MONTHS)
        )
    else:
        cleaned['arrival_month'] = np.nan
    cleaned['arrival_month_sin'] = np.sin(2 * np.pi * cleaned['arrival_month'] / 12)
    cleaned['arrival_month_cos'] = np.cos(2 * np.pi * cleaned['arrival_month'] / 12)
    cleaned['total_nights'] = cleaned['stays_in_weekend_nights'] + cleaned['stays_in_week_nights']
    cleaned['party_size'] = cleaned['adults'] + cleaned['children'] + cleaned['babies']
    cleaned['has_children'] = (
        cleaned['children'].fillna(0) + cleaned['babies'].fillna(0) > 0
    ).astype(float)
    cleaned['prior_total'] = (
        cleaned['previous_cancellations'] + cleaned['previous_bookings_not_canceled']
    )
    cleaned['prior_cancel_rate'] = (
        cleaned['previous_cancellations'] / cleaned['prior_total'].replace(0, np.nan)
    )
    cleaned['adr_per_person'] = cleaned['adr'] / cleaned['party_size'].replace(0, np.nan)
    cleaned['log_adr'] = np.log1p(cleaned['adr'].clip(lower=0))
    cleaned['log_lead_time'] = np.log1p(cleaned['lead_time'])

    arrival = pd.to_datetime(
        {
            'year': cleaned['arrival_date_year'],
            'month': cleaned['arrival_month'],
            'day': cleaned['arrival_date_day_of_month'],
        },
        errors='coerce',
    )
    booking = arrival - pd.to_timedelta(cleaned['lead_time'], unit='D')
    cleaned['arrival_ordinal'] = (arrival - pd.Timestamp('2023-01-01')).dt.days.astype(float)
    cleaned['arrival_dow'] = arrival.dt.day_of_week.astype(float)
    cleaned['arrival_day_sin'] = np.sin(2 * np.pi * cleaned['arrival_date_day_of_month'] / 31)
    cleaned['arrival_day_cos'] = np.cos(2 * np.pi * cleaned['arrival_date_day_of_month'] / 31)
    cleaned['arrival_week_sin'] = np.sin(2 * np.pi * cleaned['arrival_date_week_number'] / 53)
    cleaned['arrival_week_cos'] = np.cos(2 * np.pi * cleaned['arrival_date_week_number'] / 53)
    cleaned['booking_ordinal'] = (booking - pd.Timestamp('2023-01-01')).dt.days.astype(float)
    cleaned['booking_month'] = booking.dt.month.astype(float)
    cleaned['booking_month_sin'] = np.sin(2 * np.pi * cleaned['booking_month'] / 12)
    cleaned['booking_month_cos'] = np.cos(2 * np.pi * cleaned['booking_month'] / 12)
    cleaned['booking_dow'] = booking.dt.day_of_week.astype(float)

    # Company is intentionally excluded: it did not improve later-period validation.
    for column in (item for item in CATEGORICAL_RAW if item != 'company'):
        source = raw[column] if column in raw else pd.Series('', index=raw.index)
        cleaned[column] = _normalize_category(source, column)
    return cleaned


def predict_proba(db_path):
    """Return booking IDs and at-risk probabilities for the SQLite `features` table."""
    with sqlite3.connect(db_path) as connection:
        raw = pd.read_sql_query('SELECT * FROM features', connection)
    if 'booking_id' not in raw.columns:
        raise ValueError("The features table must contain a 'booking_id' column")
    if raw.empty:
        return pd.DataFrame({
            'booking_id': raw['booking_id'].copy(),
            'risk_score': pd.Series(dtype=float),
        })

    model_path = Path(__file__).resolve().with_name('model.joblib')
    bundle = joblib.load(model_path)
    features = _clean_features(raw).reindex(columns=bundle['feature_columns'])
    categorical = bundle['categorical_columns']
    features[categorical] = bundle['encoder'].transform(features[categorical])
    scores = bundle['model'].predict_proba(features)[:, 1]
    return pd.DataFrame({
        'booking_id': raw['booking_id'].to_numpy(copy=True),
        'risk_score': np.clip(scores.astype(float), 0.0, 1.0),
    })
