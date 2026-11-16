"""Every analysis parameter in one place. Values reproduce the manuscript exactly;
change them only deliberately and report the change."""
SEED = 42

# ---- Trip-time definitions (Section 2.3) ----
def NIGHT_HOURS(h):            # departure 20:00-05:59
    return (h >= 20) | (h < 6)
PEAK_HOURS = [7, 8, 16, 17]    # departure 07:00-08:59, 16:00-17:59

# ---- Feature groups ----
DEMO = ['Age', 'Gender', 'Education', 'BMI', 'Retired']
BINARY = ['Gender', 'Retired']
EXPOSURE = ['nTrip', 'nNightTrip', 'nPeakTrip', 'nTrip_32km']
KINEMATIC = ['n_ACC', 'n_BRK', 'n_HTURN', 'speed_mean', 'speed_sd', 'speed_max',
             'acceleration_value', 'braking_value', 'hturn_value']
VEHICLE = ['rpm_mean', 'rpm_sd', 'rpm_max', 'egl_mean', 'egl_sd', 'egl_max',
           'air_mean', 'air_sd', 'air_max', 'fuel_mean', 'fuel_sd', 'fuel_max',
           'thro_mean', 'thro_sd', 'thro_max', 'travel_time', 'dist_per_trip']
TRIP_MEAN_COLS = ['travel_time', 'dist_per_trip', 'n_ACC', 'n_BRK', 'n_HTURN',
                  'speed_mean', 'speed_sd', 'speed_max', 'rpm_mean', 'rpm_sd', 'rpm_max',
                  'acceleration_value', 'braking_value', 'hturn_value',
                  'egl_mean', 'egl_sd', 'egl_max', 'air_mean', 'air_sd', 'air_max',
                  'fuel_mean', 'fuel_sd', 'fuel_max', 'thro_mean', 'thro_sd', 'thro_max']
# 35 base features (supplementary analyses) and 36 with speed_cv (main pipeline)
BASE_FEATURES = DEMO + EXPOSURE + TRIP_MEAN_COLS
MAIN_FEATURES = BASE_FEATURES + ['speed_cv']

# ---- Multicollinearity (Stage 1): single pass, not iterative ----
VIF_THRESHOLD = 10

# ---- K-Means (Stage 2) ----
K = 4
K_RANGE_ELBOW = range(1, 11)
KMEANS_N_INIT_ELBOW = 10
KMEANS_N_INIT = 20

# ---- Random Forest (Stage 4); unlisted settings are scikit-learn defaults ----
RF_RANKING = dict(n_estimators=500, class_weight='balanced', random_state=SEED, n_jobs=-1)
RF_LOOCV = dict(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1)

# ---- Welch t-tests (Stage 5) ----
FDR_Q = 0.05

# ---- Logistic regression (Stage 6) ----
# L2 (ridge) regularization (liblinear default penalty).
# C = 0.1 was fixed, not tuned.
LR = dict(C=0.1, class_weight='balanced', solver='liblinear',
          max_iter=1000, random_state=SEED)
SMOTE_K_MAX = 4          # k = min(4, n_minority_in_fold - 1)
THRESHOLD = 0.5

# ---- Uncertainty ----
N_BOOT = 2000
N_PERM = 1000

# ---- Supplementary analyses ----
TEMPORAL_QUANTILE = 0.65      # month-index quantile -> cutoff = end of Dec 2022
COLD_START_TRIPS = [10, 25, 50, 100, 200, None]
AGE_BANDS = dict(bins=[64, 72, 79, 120], labels=['65-72', '73-79', '80+'])
