"""Loading raw trips and aggregating to one row per participant (Section 2.3).

Aggregation rules (what the code does, stated for the Methods):
  * Demographics: first record per participant.
  * nTrip, nNightTrip, nPeakTrip, nTrip_32km: totals over the observation window.
  * All other trip-level indices: mean over trips, skipping missing values
    (hard-event severities are therefore means over trips with >= 1 event).
  * speed_cv = participant-mean speed SD / participant-mean speed.
  * Participant-level missing values are filled with the cohort median
    (label-free); no outlier capping or other transformation is applied.
"""
import numpy as np
import pandas as pd
from . import config as C


def load_trips(path):
    df = pd.read_csv(path, dtype={'participant_id': str}, low_memory=False)
    df['thro_max'] = pd.to_numeric(df['thro_max'], errors='coerce')   # 'NaN ' strings
    df['trip_start'] = pd.to_datetime(df['trip start'])
    df['trip_end'] = pd.to_datetime(df['trip end'])
    h = df['trip_start'].dt.hour
    df['hour'] = h
    df['is_night'] = C.NIGHT_HOURS(h).astype(int)
    df['is_peak'] = h.isin(C.PEAK_HOURS).astype(int)
    return df


def aggregate(trips, with_speed_cv=True):
    g = trips.groupby('participant_id')
    out = {c: g[c].first() for c in C.DEMO}
    out['MCI'] = g['MCI'].first()
    out['nTrip'] = g['MCI'].count()
    out['nNightTrip'] = g['is_night'].sum()
    out['nPeakTrip'] = g['is_peak'].sum()
    out['nTrip_32km'] = g['travel_32km'].sum()
    for c in C.TRIP_MEAN_COLS:
        if c in trips.columns:
            out[c] = g[c].mean()
    df = pd.DataFrame(out).reset_index()
    if with_speed_cv:
        df['speed_cv'] = df['speed_sd'] / df['speed_mean'].replace(0, np.nan)
    return df


def impute_median(df, feats):
    """Cohort-median fill for participant-level gaps (label-free)."""
    return df[feats].fillna(df[feats].median())


def describe(trips, dfp):
    return {
        'participants': len(dfp), 'MCI': int(dfp.MCI.sum()), 'trips': len(trips),
        'age_mean': round(dfp.Age.mean(), 1), 'age_sd': round(dfp.Age.std(), 1),
        'age_range': f"{dfp.Age.min()}-{dfp.Age.max()}", 'male_pct': round(100 * dfp.Gender.mean(), 1),
        'trips_median': dfp.nTrip.median(),
        'trips_iqr': f"{dfp.nTrip.quantile(.25)}-{dfp.nTrip.quantile(.75)}",
        'night_trips': int(trips.is_night.sum()), 'night_pct': round(100 * trips.is_night.mean(), 1),
        'peak_trips': int(trips.is_peak.sum()), 'peak_pct': round(100 * trips.is_peak.mean(), 1),
        'MCI_trips': int(trips.MCI.sum()), 'MCI_trip_pct': round(100 * trips.MCI.mean(), 1),
        'first_trip': str(trips.trip_start.min()), 'last_trip': str(trips.trip_start.max()),
        'participants_missing_air': int(dfp.air_sd.isna().sum()),
    }
