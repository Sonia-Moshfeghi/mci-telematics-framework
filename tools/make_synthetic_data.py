"""Generate a synthetic trip-level CSV with the same schema as the study data, so the
code can be run end-to-end without access to participant data.  Values are random draws
from hard-coded plausible ranges; they carry NO information about the real cohort and
results computed on them are meaningless.

    python tools/make_synthetic_data.py --out data/synthetic_df.csv
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='data/synthetic_df.csv')
ap.add_argument('--participants', type=int, default=51)
ap.add_argument('--n-mci', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

rows = []
for i in range(a.participants):
    pid = str(9000 + i)
    demo = dict(Age=int(rng.integers(65, 88)), Gender=int(rng.integers(0, 2)),
                Education=int(rng.integers(0, 11)), BMI=int(rng.integers(0, 5)),
                Retired=int(rng.random() < .8), MCI=int(i < a.n_mci))
    has_air = rng.random() > .3
    start = pd.Timestamp('2021-06-15') + pd.Timedelta(days=int(rng.integers(0, 60)))
    for _ in range(int(rng.integers(60, 200))):
        start += pd.Timedelta(hours=float(rng.exponential(60)))
        dur = float(rng.gamma(2, 600)); dist = dur / 3600 * rng.uniform(15, 60)
        sp_mean = rng.uniform(15, 55); sp_sd = sp_mean * rng.uniform(.4, .9)
        n_acc, n_brk, n_turn = rng.poisson([.8, 1.1, 1.3])
        th = rng.uniform(12, 28)
        rows.append(dict(participant_id=pid,
            **{'trip start': start.strftime('%m/%d/%Y %H:%M'),
               'trip end': (start + pd.Timedelta(seconds=dur)).strftime('%m/%d/%Y %H:%M')},
            **{k: v for k, v in demo.items() if k != 'MCI'},
            travel_32km=int(dist > 32), travel_time=dur, dist_per_trip=dist,
            n_ACC=n_acc, n_BRK=n_brk, n_HTURN=n_turn,
            speed_mean=sp_mean, speed_sd=sp_sd, speed_max=sp_mean + 2.5 * sp_sd,
            rpm_mean=rng.uniform(1100, 1800), rpm_sd=rng.uniform(300, 700), rpm_max=rng.uniform(2000, 4000),
            acceleration_value=rng.uniform(.3, .45) if n_acc else np.nan,
            braking_value=-rng.uniform(.3, .45) if n_brk else np.nan,
            hturn_value=rng.uniform(.3, .45) if n_turn else np.nan,
            egl_mean=rng.uniform(25, 50), egl_sd=rng.uniform(15, 30), egl_max=100.0,
            air_mean=rng.uniform(20, 35) if has_air else np.nan,
            air_sd=rng.uniform(0, 2) if has_air else np.nan,
            air_max=rng.uniform(25, 40) if has_air else np.nan,
            fuel_mean=rng.uniform(20, 90), fuel_sd=rng.uniform(0, 8), fuel_max=rng.uniform(25, 100),
            thro_mean=th, thro_sd=th * rng.uniform(.2, .5), thro_max=th * rng.uniform(2, 4),
            MCI=demo['MCI']))
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f'wrote {len(rows)} synthetic trips for {a.participants} participants to {a.out}')
