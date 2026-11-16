"""Fig. 10 (within-subject temporal stability), Fig. 11 (subgroup performance) and
Fig. 12 (predicted probabilities by gender and equalized-odds gaps).
An AUC is reported only where a subgroup contains at least two MCI and two non-MCI participants."""
import os
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.preprocessing import StandardScaler
from _common import args, load, savefig, plt, FIGS
from mci_telematics import config as C, data, features, models

a = args()
trips, dfp = load(a.data, with_speed_cv=False)
y = dfp.MCI.values.astype(int)
REDUCED = features.vif_reduce(dfp, C.BASE_FEATURES)
MIN_CASES = 2


def both(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.pdf'), bbox_inches='tight'); savefig(fig, name + '.png')


# ---------------- Fig. 10 ----------------
month = (trips.trip_start.dt.year - trips.trip_start.dt.year.min()) * 12 + trips.trip_start.dt.month
cut = month.quantile(C.TEMPORAL_QUANTILE)
early, late = trips[month <= cut], trips[month > cut]
common = sorted(set(early.participant_id) & set(late.participant_id))
tr = data.aggregate(early[early.participant_id.isin(common)], with_speed_cv=False)
te = data.aggregate(late[late.participant_id.isin(common)], with_speed_cv=False)
uf = [f for f in REDUCED if tr[f].std() > 0]
sc = StandardScaler(); Xtr, ytr = models._smote(sc.fit_transform(data.impute_median(tr, uf).values), tr.MCI.values.astype(int))
pt = LogisticRegression(**C.LR).fit(Xtr, ytr).predict_proba(sc.transform(data.impute_median(te, uf).values))[:, 1]
yt = te.MCI.values.astype(int)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
per = trips.trip_start.dt.to_period('M')
cnt = trips.groupby(per).size(); cutp = early.trip_start.max().to_period('M')
ax[0].bar(range(len(cnt)), cnt.values, width=1.0, color=['#4C72B0' if p <= cutp else '#C44E52' for p in cnt.index], alpha=0.85)
ax[0].set_xticks(range(0, len(cnt), 4)); ax[0].set_xticklabels([str(p) for p in cnt.index[::4]], rotation=30, ha='right', fontsize=8)
ax[0].bar([0], [0], color='#4C72B0', label='Training segment (to 31 Dec 2022)'); ax[0].bar([0], [0], color='#C44E52', label='Test segment (from 1 Jan 2023)')
ax[0].set(xlabel='Month', ylabel='Trips', title='Within-subject temporal split'); ax[0].legend(fontsize=9)
ax[1].scatter(np.where(yt == 0)[0], pt[yt == 0], s=45, color='#4C72B0', label='Non-MCI')
ax[1].scatter(np.where(yt == 1)[0], pt[yt == 1], s=45, color='#C44E52', label='MCI')
ax[1].axhline(C.THRESHOLD, ls='--', c='gray', lw=1)
ax[1].set(xlabel='Participant (same 14 participants as in the training segment)', ylabel='Predicted $P$(MCI), later trips',
          title=f'Later-segment predictions (n = {len(yt)}, MCI = {int(yt.sum())})'); ax[1].legend(fontsize=9)
fig.tight_layout(); both(fig, 'fig10_temporal_stability')

# ---------------- Fig. 11 ----------------
p, d = models.loocv(dfp, REDUCED)
age = pd.cut(dfp.Age, **C.AGE_BANDS)


def rates(mask):
    tn, fp, fn, tp = confusion_matrix(y[mask], d[mask], labels=[0, 1]).ravel()
    div = lambda u, v: u / v if v else np.nan
    return dict(sens=div(tp, tp + fn), spec=div(tn, tn + fp), FPR=div(fp, fp + tn), FNR=div(fn, fn + tp))


def estimable(mask):
    return y[mask].sum() >= MIN_CASES and (1 - y[mask]).sum() >= MIN_CASES


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (g, name, col) in enumerate([(0, 'Female', '#C44E52'), (1, 'Male', '#4C72B0')]):
    m = (dfp.Gender == g).values; ax = axes[0][i]
    ax.set_title(f'{name} (n = {m.sum()}, MCI = {y[m].sum()})')
    if estimable(m):
        f, t, _ = roc_curve(y[m], p[m]); ax.plot(f, t, lw=2.5, color=col, label=f'AUC = {roc_auc_score(y[m], p[m]):.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3); ax.legend(fontsize=9); ax.set(xlabel='FPR', ylabel='TPR')
    else:
        rank = int((p[m][y[m] == 0] < p[m][y[m] == 1][0]).sum()) if y[m].sum() == 1 else None
        ax.hist(p[m][y[m] == 0], bins=10, range=(0, 1), color=col, alpha=0.6, label='Non-MCI')
        for v in p[m][y[m] == 1]:
            ax.axvline(v, color='#C44E52', lw=2.5, label='MCI participant')
        ax.axvline(C.THRESHOLD, color='gray', ls='--', lw=1)
        ax.set(xlabel='Predicted $P$(MCI)', ylabel='Participants'); ax.legend(fontsize=9)
        ax.text(0.97, 0.75, f'AUC not estimable\n(single MCI case; ranked above\n{rank} of {int((1 - y[m]).sum())} non-MCI participants)',
                ha='right', va='top', transform=ax.transAxes, fontsize=9)
ax = axes[0][2]; R = [rates((dfp.Gender == g).values) for g in (0, 1)]; x = np.array([0, 1])
ax.bar(x - 0.15, [r['FPR'] for r in R], width=0.3, color='#DD8452', alpha=0.85, label='FPR')
ax.bar(x + 0.15, [r['FNR'] for r in R], width=0.3, color='#C44E52', alpha=0.85, label='FNR')
for xi, r in enumerate(R):
    ax.text(xi - 0.15, r['FPR'] + 0.02, f"{r['FPR']:.2f}", ha='center', fontsize=9); ax.text(xi + 0.15, r['FNR'] + 0.02, f"{r['FNR']:.2f}", ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(['Female', 'Male (FNR from 1 MCI case)']); ax.set(ylabel='Error rate', title='FPR and FNR by gender (descriptive)'); ax.legend(fontsize=9)
labels = C.AGE_BANDS['labels']; masks = [(age == l).values for l in labels]
ax = axes[1][0]
for i, (l, m) in enumerate(zip(labels, masks)):
    if estimable(m):
        v = roc_auc_score(y[m], p[m]); ax.bar(i, v, width=0.5, color=['#4C72B0', '#55A868', '#C44E52'][i]); ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=9)
    else:
        ax.text(i, 0.05, 'not estimable\n(no MCI case)', ha='center', fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.axhline(0.5, color='gray', ls='--', alpha=0.5); ax.set_ylim(0, 1); ax.set(ylabel='AUC', title='AUC by age group')
ax = axes[1][1]; R = [rates(m) for m in masks]; x = np.arange(3)
ax.bar(x - 0.2, [0 if np.isnan(r['sens']) else r['sens'] for r in R], width=0.35, color='#C44E52', alpha=0.85, label='Sensitivity')
ax.bar(x + 0.2, [r['spec'] for r in R], width=0.35, color='#4C72B0', alpha=0.85, label='Specificity')
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set(ylabel='Score', title='Sensitivity and specificity by age group'); ax.legend(fontsize=9)
ax = axes[1][2]; fc, mp = calibration_curve(y, p, n_bins=5, strategy='quantile')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1, label='Perfect'); ax.plot(mp, fc, 'o-', lw=2.5, color='#4C72B0', label=f'LR (Brier = {brier_score_loss(y, p):.3f})')
ax.set(xlabel='Mean predicted probability', ylabel='Observed MCI fraction', title='Overall calibration'); ax.legend(fontsize=9)
fig.tight_layout(); both(fig, 'fig11_subgroups')

# ---------------- Fig. 12 ----------------
TEL = features.vif_reduce(dfp, C.EXPOSURE + C.KINEMATIC + C.VEHICLE)
p_tel, d_tel = models.loocv(dfp, TEL)


def gaps(pred):
    r = {}
    for g in (0, 1):
        m = (dfp.Gender == g).values
        tn, fp, fn, tp = confusion_matrix(y[m], pred[m], labels=[0, 1]).ravel()
        r[g] = (fp / (fp + tn), fn / (fn + tp))
    return abs(r[0][0] - r[1][0]), abs(r[0][1] - r[1][1])


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for i, (g, name) in enumerate([(0, 'Female'), (1, 'Male')]):
    m = (dfp.Gender == g).values
    ax[i].hist(p[m][y[m] == 0], bins=10, range=(0, 1), color='#4C72B0', alpha=0.7, label=f'Non-MCI (n = {int((1 - y[m]).sum())})')
    ax[i].hist(p[m][y[m] == 1], bins=10, range=(0, 1), color='#C44E52', alpha=0.7, label=f'MCI (n = {int(y[m].sum())})')
    ax[i].axvline(C.THRESHOLD, color='k', ls='--', lw=1)
    ax[i].set(xlabel='Predicted $P$(MCI)', ylabel='Participants', title=f'{name} (n = {m.sum()}, MCI = {y[m].sum()})'); ax[i].legend(fontsize=9)
G = [gaps(d), gaps(d_tel)]; x = np.array([0, 1])
ax[2].bar(x - 0.17, [v[0] for v in G], width=0.34, color='#DD8452', label=r'$|\Delta$FPR$|$')
ax[2].bar(x + 0.17, [v[1] for v in G], width=0.34, color='#C44E52', label=r'$|\Delta$FNR$|$')
for xi, v in enumerate(G):
    ax[2].text(xi - 0.17, v[0] + 0.02, f'{v[0]:.2f}', ha='center', fontsize=9); ax[2].text(xi + 0.17, v[1] + 0.02, f'{v[1]:.2f}', ha='center', fontsize=9)
ax[2].set_xticks(x); ax[2].set_xticklabels(['Demographics + Telematics', 'Telematics only'])
ax[2].set(ylabel='Equalized-odds gap', title='Equalized-odds gaps by gender (descriptive)'); ax[2].legend(fontsize=9)
fig.tight_layout(); both(fig, 'fig12_equalized_odds')
