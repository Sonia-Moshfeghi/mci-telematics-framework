"""Sensitivity analyses of the main model (Appendix B, Tables B1 and B2).
 * class-imbalance handling: SMOTE (primary), class weights only, SMOTE-NC
 * median imputation and VIF screening recomputed strictly within each training fold
 * exclusion of the ambient-temperature covariate air_sd"""
from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from _common import args, load, save, save_json
from mci_telematics import config as C, data, features, models

a = args()
trips, dfp = load(a.data)
y = dfp.MCI.values.astype(int)
TEL = C.EXPOSURE + C.KINEMATIC + C.VEHICLE
FULL = features.vif_reduce(dfp, C.MAIN_FEATURES)
S = {'Age + Gender': ['Age', 'Gender'],
     'Demographics only': features.vif_reduce(dfp, C.DEMO),
     'Telematics only': features.vif_reduce(dfp, TEL),
     'Demographics + Telematics': FULL,
     'No Gender': [f for f in FULL if f != 'Gender']}
S['Age + Gender + Telematics'] = features.vif_reduce(dfp, ['Age', 'Gender'] + S['Telematics only'])


def fit_predict(Xa, ya, Xb, feats, mode):
    """Scale on the training fold, balance classes as requested, fit, predict the held-out row."""
    sc = StandardScaler(); Za = sc.fit_transform(Xa)
    cat = [i for i, f in enumerate(feats) if f in C.BINARY]
    k = min(C.SMOTE_K_MAX, int(ya.sum()) - 1)
    if mode == 'smote':
        Za, ya = SMOTE(random_state=C.SEED, k_neighbors=k).fit_resample(Za, ya)
    elif mode == 'smotenc':          # binary indicators resampled as categories, never interpolated
        Za, ya = SMOTENC(categorical_features=cat, random_state=C.SEED, k_neighbors=k).fit_resample(Za, ya)
    m = LogisticRegression(**C.LR).fit(Za, ya)
    zb = sc.transform(Xb)
    return m.predict_proba(zb)[:, 1], m.predict(zb)


def loocv_mode(feats, mode):
    X = data.impute_median(dfp, feats).values
    p, d = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        p[te], d[te] = fit_predict(X[tr], y[tr], X[te], feats, mode)
    return p, d


print('Class-imbalance handling (Table B2)')
rows = []
for name, feats in S.items():
    for mode in ['smote', 'weights_only', 'smotenc']:
        if mode == 'smotenc' and not any(f in C.BINARY for f in feats):
            continue                  # no binary indicator in this feature set
        p, d = loocv_mode(feats, mode)
        rows.append(dict(models.metrics(y, p, d, name), imbalance=mode, n_features=len(feats)))
save(pd.DataFrame(rows), 'tableB2_class_imbalance.csv')

# share of SMOTE-generated observations with a non-binary value of Gender / Retired
frac = []
X = data.impute_median(dfp, FULL).values
for tr, te in LeaveOneOut().split(X):
    Z = StandardScaler().fit_transform(X[tr])
    Zs, _ = SMOTE(random_state=C.SEED, k_neighbors=min(C.SMOTE_K_MAX, int(y[tr].sum()) - 1)).fit_resample(Z, y[tr])
    syn = Zs[len(tr):]
    for b in C.BINARY:
        j = FULL.index(b)
        frac.append((b, np.mean(~np.isin(np.round(syn[:, j], 6), np.round(np.unique(Z[:, j]), 6)))))
frac = pd.DataFrame(frac, columns=['feature', 'share_intermediate']).groupby('feature').share_intermediate.mean()
save_json(frac.to_dict(), 'sA_smote_intermediate_binary_values.json')

print('Imputation and VIF screening within each training fold (Table B1)')
main = []
p, d = models.loocv(dfp, FULL); main.append(dict(models.metrics(y, p, d, 'Primary analysis'), n_features=len(FULL)))
p, d = loocv_mode(FULL, 'weights_only'); main.append(dict(models.metrics(y, p, d, 'Class weights only (no SMOTE)'), n_features=len(FULL)))
p, d = loocv_mode(FULL, 'smotenc'); main.append(dict(models.metrics(y, p, d, 'SMOTE-NC'), n_features=len(FULL)))

R = dfp[FULL]; p, d = np.zeros(len(y)), np.zeros(len(y))
for tr, te in LeaveOneOut().split(R):
    med = R.iloc[tr].median()
    p[te], d[te] = fit_predict(R.iloc[tr].fillna(med).values, y[tr], R.iloc[te].fillna(med).values, FULL, 'smote')
main.append(dict(models.metrics(y, p, d, 'Median imputation within folds'), n_features=len(FULL)))

R = dfp[C.MAIN_FEATURES]; p, d, selected = np.zeros(len(y)), np.zeros(len(y)), []
for tr, te in LeaveOneOut().split(R):
    med = R.iloc[tr].median(); Xa, Xb = R.iloc[tr].fillna(med), R.iloc[te].fillna(med)
    cols = [c for c in C.MAIN_FEATURES if Xa[c].std() > 0]
    Z = StandardScaler().fit_transform(Xa[cols].values)
    keep = [c for i, c in enumerate(cols) if variance_inflation_factor(Z, i) <= C.VIF_THRESHOLD]
    selected.append(tuple(keep))
    p[te], d[te] = fit_predict(Xa[keep].values, y[tr], Xb[keep].values, keep, 'smote')
main.append(dict(models.metrics(y, p, d, 'Median imputation and VIF screening within folds'),
                 n_features=float(np.mean([len(s) for s in selected]))))
save_json(dict(folds_with_same_set_as_full_cohort=sum(set(s) == set(FULL) for s in selected),
               n_folds=len(selected), set_sizes=dict(Counter(len(s) for s in selected)),
               folds_retaining_each_feature=dict(Counter(f for s in selected for f in s))),
          'sA_infold_vif_selection.json')

no_air = [f for f in FULL if f != 'air_sd']
p, d = models.loocv(dfp, no_air); main.append(dict(models.metrics(y, p, d, 'Excluding air_sd'), n_features=len(no_air)))
save(pd.DataFrame(main), 'tableB1_main_model_sensitivity.csv')
save_json(dfp.groupby('MCI').air_sd.apply(lambda s: int(s.isna().sum())).to_dict(), 'sA_air_sd_missing_by_group.json')
