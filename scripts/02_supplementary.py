"""Supplementary analyses, manuscript Section 3.8 (Tables 7-9, Figs. 9-12).
Uses the 35 base features (all indices except speed_cv)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from _common import args, load, save, save_json, savefig, plt
from mci_telematics import config as C, data, features, clustering, models

a = args()
trips, dfp = load(a.data, with_speed_cv=False)
y = dfp.MCI.values.astype(int)
F = C.BASE_FEATURES
TEL = C.EXPOSURE + C.KINEMATIC + C.VEHICLE
REDUCED = features.vif_reduce(dfp, F)


def run(feats, label, df=dfp):
    p, d = models.loocv(df, feats)
    return dict(models.metrics(df.MCI.values.astype(int), p, d, label), n_features=len(feats)), p, d


print('3.8.1 Mixed-type clustering sensitivity')
sil, ext, _ = clustering.mixed_type_sensitivity(dfp, F)
save(sil, 's3_8_1_silhouette.csv'); save(ext, 's3_8_1_external_chi2_K4.csv')
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in sil.columns[1:]:
    ax.plot(sil.K, sil[c], 'o-', label=c)
ax.axvline(C.K, ls='--', c='k'); ax.set(xlabel='K', ylabel='Silhouette'); ax.legend(); savefig(fig, 'fig9_clustering_sensitivity.png')

print('3.8.2 Ablation (Table 7, core rows)')
sets7 = {'Demographics only': features.vif_reduce(dfp, C.DEMO),
         'Telematics only': features.vif_reduce(dfp, TEL),
         'Demographics + Telematics': features.vif_reduce(dfp, C.DEMO + TEL)}
sets7['No Gender (sensitivity)'] = [f for f in sets7['Demographics + Telematics'] if f != 'Gender']
t7 = pd.DataFrame([run(f, k)[0] for k, f in sets7.items()])
save(t7, 'table7_ablation.csv'); save_json(sets7, 'table7_feature_sets.json')

print('3.8.3 Incremental feature sets (Table 8)')
sets8 = {'Exposure only': C.EXPOSURE, 'Exposure + Kinematics': C.EXPOSURE + C.KINEMATIC,
         'Exposure + Kin. + Vehicle': TEL, 'Full (+ Demographics)': C.DEMO + TEL}
sets8 = {k: features.vif_reduce(dfp, v) for k, v in sets8.items()}
save(pd.DataFrame([run(f, k)[0] for k, f in sets8.items()]), 'table8_incremental.csv')
save_json(sets8, 'table8_feature_sets.json')

print('3.8.4 Temporal split (same participants in both segments -> within-subject stability)')
month = (trips.trip_start.dt.year - trips.trip_start.dt.year.min()) * 12 + trips.trip_start.dt.month
cut = month.quantile(C.TEMPORAL_QUANTILE)
early, late = trips[month <= cut], trips[month > cut]
common = sorted(set(early.participant_id) & set(late.participant_id))
tr = data.aggregate(early[early.participant_id.isin(common)], with_speed_cv=False)
te = data.aggregate(late[late.participant_id.isin(common)], with_speed_cv=False)
uf = [f for f in REDUCED if tr[f].std() > 0]
sc = StandardScaler(); Xtr = sc.fit_transform(data.impute_median(tr, uf).values)
Xtr, ytr = models._smote(Xtr, tr.MCI.values.astype(int))
clf = LogisticRegression(**C.LR).fit(Xtr, ytr)
Xte = sc.transform(data.impute_median(te, uf).values)
pt, dt = clf.predict_proba(Xte)[:, 1], clf.predict(Xte)
save(pd.DataFrame([models.metrics(te.MCI.values.astype(int), pt, dt, 'Temporal hold-out', ci=False)]), 's3_8_4_temporal.csv')
save(pd.DataFrame(dict(participant_id=te.participant_id, MCI=te.MCI, prob=pt)), 's3_8_4_temporal_predictions.csv')
save_json(dict(month_index_cutoff=cut, train_period=f'{early.trip_start.min()} to {early.trip_start.max()}',
               test_period=f'{late.trip_start.min()} to {late.trip_start.max()}',
               n_participants=len(common), n_MCI=int(te.MCI.sum())), 's3_8_4_temporal_split.json')

print('Cold-start (Section 4.5): first N trips; all non-constant indices, no VIF step')
rows = []
for n in C.COLD_START_TRIPS:
    sub = trips.sort_values('trip_start').groupby('participant_id').head(n) if n else trips
    g = sub.groupby('participant_id')
    cd = pd.DataFrame({c: g[c].first() for c in C.DEMO})
    cd['MCI'] = g.MCI.first(); cd['nNightTrip'] = g.is_night.sum(); cd['nTrip_32km'] = g.travel_32km.sum()
    cd['nTrip'] = g.size()
    for c in C.TRIP_MEAN_COLS:
        cd[c] = g[c].mean()
    cd = cd.reset_index()
    cf = [f for f in F if f in cd.columns and cd[f].std() > 0]   # nPeakTrip not computed here
    m, _, _ = run(cf, f'{n} trips' if n else 'All trips', df=cd)
    rows.append(m)
save(pd.DataFrame(rows), 's4_5_cold_start.csv')

print('3.8.5 Subgroups and fairness (Table 9, Figs. 11-12)')
p, d = models.loocv(dfp, REDUCED)
dfp['age_band'] = pd.cut(dfp.Age, **C.AGE_BANDS)
sub = []
for col, levels in [('Gender', [(0, 'Female'), (1, 'Male')]),
                    ('age_band', [(b, b) for b in C.AGE_BANDS['labels']])]:
    for v, name in levels:
        m = (dfp[col] == v).values
        r = models.metrics(y[m], p[m], d[m], name, ci=False)
        r.update(n=int(m.sum()), MCI=int(y[m].sum()), FPR=1 - r['specificity'], FNR=1 - r['sensitivity'])
        sub.append(r)
save(pd.DataFrame(sub), 'table9_subgroups.csv')
eo = []
for label, feats in [('With demographics', REDUCED), ('Telematics only', sets7['Telematics only'])]:
    pp, dd = models.loocv(dfp, feats)
    g = {}
    for v, name in [(0, 'Female'), (1, 'Male')]:
        m = (dfp.Gender == v).values
        tn, fp, fn, tp = confusion_matrix(y[m], dd[m], labels=[0, 1]).ravel()
        g[name] = dict(FPR=fp / (fp + tn), FNR=fn / (fn + tp),
                       AUC=roc_auc_score(y[m], pp[m]) if len(set(y[m])) == 2 else np.nan)
    eo.append(dict(model=label, dFPR=abs(g['Female']['FPR'] - g['Male']['FPR']),
                   dFNR=abs(g['Female']['FNR'] - g['Male']['FNR']),
                   AUC_female=g['Female']['AUC'], AUC_male=g['Male']['AUC']))
eo = pd.DataFrame(eo); eo['total'] = eo.dFPR + eo.dFNR
save(eo, 's3_8_5_equalized_odds.csv')
