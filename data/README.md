# Data

The analysis expects one trip-level CSV (`df.csv`, one row per trip) with these columns:

| Column(s) | Content |
|---|---|
| `participant_id` | Study ID (string); one ID per participant |
| `trip start`, `trip end` | Timestamps (`M/D/YYYY H:MM`) |
| `Age`, `Gender` (0 = female, 1 = male), `Education` (0-10), `BMI` (category 0-4), `Retired` (0/1) | Demographics, constant within participant |
| `travel_32km` | 1 if trip distance > 32 km |
| `travel_time` (s), `dist_per_trip` (km) | Trip duration and distance |
| `n_ACC`, `n_BRK`, `n_HTURN` | Hard acceleration / braking / turning events in the trip (\|a\| > 0.3 g) |
| `acceleration_value`, `braking_value`, `hturn_value` | Mean magnitude of those events (m s⁻²); empty when the trip had no event |
| `speed_*`, `rpm_*`, `egl_*`, `air_*`, `fuel_*`, `thro_*` (`_mean`, `_sd`, `_max`) | Within-trip summaries of OBD-II channels |
| `MCI` | Diagnostic label (1 = MCI), constant within participant |

**Participant data are not included in this package and cannot be shared.**

To run the code without them:

```bash
python tools/make_synthetic_data.py --out data/synthetic_df.csv
bash run_all.sh data/synthetic_df.csv 20
```

Synthetic values are random and results computed on them are meaningless; they only
demonstrate that the pipeline runs.
