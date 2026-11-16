"""Main pipeline, manuscript Sections 3.1-3.7 (Stages 1-6 of Fig. 2).
Tables 4, 5, 6 and Figs. 3, 4, 5, 6, 7, 8."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from _common import args, load, save, save_json, savefig, plt
from mci_telematics import config as C, data, features, clustering, stats_tests, models, interpret

a = args()
trips, dfp = load(a.data)
y = dfp.MCI.values.astype(int)
F = C.MAIN_FEATURES

print('3.1 Dataset characteristics')
save_json(data.describe(trips, dfp), 's3_1_dataset.json')

print('3.2 Stage 1 - VIF (single pass, threshold 10)')
vif = features.vif_table(dfp, F); save(vif, 's3_2_vif_all_features.csv')
REDUCED = features.vif_reduce(dfp, F)
print(f'  retained {len(REDUCED)}: {REDUCED}')

print('3.3 Stage 2 - K-Means (K=4, labels withheld)')
Xs = clustering.standardized(dfp, F)
inertia = clustering.elbow(Xs)
dfp['cluster'] = clustering.kmeans(Xs)
chi2, p_chi, dof = clustering.chi2_vs_mci(dfp.cluster, dfp.MCI)
prof = dfp.groupby('cluster')[['MCI', 'nTrip', 'nNightTrip', 'nPeakTrip', 'speed_mean', 'thro_mean', 'thro_sd']].mean()
prof.insert(0, 'n', dfp.groupby('cluster').size()); prof['MCI'] *= 100
save(prof.rename(columns={'MCI': 'MCI_pct'}), 'table4_cluster_centroids.csv', index=True)
save_json(dict(chi2=chi2, p=p_chi, dof=dof), 's3_3_cluster_vs_mci_chi2.json')
pc, evr = clustering.pca2(Xs)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(list(C.K_RANGE_ELBOW), inertia, 'o-'); ax[0].axvline(C.K, ls='--', c='r')
ax[0].set(xlabel='Number of clusters K', ylabel='Within-cluster SSE', title='Elbow method')
for k in range(C.K):
    m = dfp.cluster.values == k
    ax[1].scatter(pc[m, 0], pc[m, 1], s=40, label=f'Cluster {k} (n={m.sum()})')
ax[1].scatter(pc[y == 1, 0], pc[y == 1, 1], s=160, facecolors='none', edgecolors='k', label='MCI (post hoc)')
ax[1].set(xlabel=f'PC1 ({evr[0]:.1%})', ylabel=f'PC2 ({evr[1]:.1%})', title='K-Means clusters (PCA)'); ax[1].legend(fontsize=8)
savefig(fig, 'fig3_kmeans.png')

print('3.4 Stage 3 - ANOVA + Tukey across clusters')
anova, excluded = stats_tests.anova_by_cluster(dfp, F)
save(anova, 's3_4_anova.csv'); save_json(dict(excluded_insufficient_group_size=excluded), 's3_4_anova_excluded.json')
top3 = anova.feature.head(3).tolist()
save(pd.concat([stats_tests.tukey(dfp, f) for f in top3]), 's3_4_tukey_top3.csv')
t15 = anova.head(15)[::-1]
fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(t15.feature, t15.eta2, color='#C44E52'); ax.axvline(.14, ls='--', c='k')
ax.set(xlabel='Effect size $\\eta^2$', title='ANOVA effect sizes across clusters (top 15)'); savefig(fig, 'fig4_anova_eta2.png')
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, f in zip(axs, top3):
    ax.boxplot([dfp.loc[dfp.cluster == k, f] for k in range(C.K)])
    ax.set_xticklabels([f'C{k}\n(n={(dfp.cluster == k).sum()})' for k in range(C.K)]); ax.set_title(f)
savefig(fig, 'fig5_tukey_top3.png')

print('3.5 Stage 4 - Random Forest ranking + LOOCV')
rf, imp = interpret.rf_ranking(dfp, F)
save(imp.rename('gini_importance').to_frame(), 'fig6_rf_importance.csv', index=True)
throttle = ['thro_mean', 'thro_sd', 'thro_max']
cats = {'kinematics': [f for f in C.KINEMATIC] + ['speed_cv'], 'throttle': throttle,
        'vehicle_excl_throttle': [f for f in C.VEHICLE if f not in throttle and f not in ('travel_time', 'dist_per_trip')],
        'exposure': C.EXPOSURE + ['travel_time', 'dist_per_trip'], 'demographics': C.DEMO}
save_json({k: 100 * imp[v].sum() for k, v in cats.items()}, 's3_5_rf_importance_by_category_pct.json')
p_rf, d_rf = models.loocv(dfp, F, model='rf')
save(pd.DataFrame([models.metrics(y, p_rf, d_rf, 'Random Forest LOOCV')]), 's3_5_rf_loocv.csv')
t20 = imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(t20.index, t20.values, color='#55A868')
ax.set(xlabel='Mean decrease in Gini impurity', title='Random Forest feature importance (n=51)'); savefig(fig, 'fig6_rf_importance.png')

print('3.6 Stage 5 - Welch t-tests + BH-FDR')
welch = stats_tests.welch_fdr(dfp, F)
save(welch, 'table5_welch_all.csv'); save(welch.head(10), 'table5_welch_top10.csv')

print('3.7 Stage 6 - L2-regularized logistic regression, LOOCV')
p, d = models.loocv(dfp, REDUCED)
m = models.metrics(y, p, d, 'Logistic regression (11 VIF-selected features)')
save(pd.DataFrame([m]), 'table6_lr_loocv.csv')
cal = models.calibration(y, p); save_json(cal, 's3_7_calibration.json')
X = data.impute_median(dfp, REDUCED).values
full = LogisticRegression(**C.LR).fit(StandardScaler().fit_transform(X), y)
coef = pd.DataFrame(dict(feature=REDUCED, coefficient=full.coef_[0], odds_ratio=np.exp(full.coef_[0])))
save(coef.sort_values('odds_ratio', ascending=False), 's3_7_lr_coefficients_full_fit.csv')
save(pd.DataFrame(dict(participant_id=dfp.participant_id, MCI=y, Gender=dfp.Gender, Age=dfp.Age,
                       prob=p, pred=d)), 'loocv_predictions_full_model.csv')

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ConfusionMatrixDisplay.from_predictions(y, d, display_labels=['Non-MCI', 'MCI'], ax=ax[0], colorbar=False, cmap='Blues')
fpr, tpr, _ = roc_curve(y, p); ax[1].plot(fpr, tpr, label=f'AUC={m["AUC"]:.3f}'); ax[1].plot([0, 1], [0, 1], 'k--'); ax[1].legend()
ax[1].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC (LOOCV)')
c = coef.sort_values('odds_ratio'); ax[2].barh(c.feature, c.odds_ratio, color=['#C44E52' if v > 1 else '#4C72B0' for v in c.odds_ratio])
ax[2].axvline(1, ls='--', c='k'); ax[2].set(xlabel='Odds ratio (standardized)', title='Full-data coefficients')
savefig(fig, 'fig8_lr_results.png')
from mci_telematics.plot_calibration import calibration_figure
fig, _ = calibration_figure(y, p, cal, plt)
fig.savefig(__import__('os').path.join(__import__('_common').FIGS, 'fig7_calibration.pdf'), bbox_inches='tight')
savefig(fig, 'fig7_calibration.png')
print(f"  AUC={m['AUC']:.3f} [{m['CI_lo']:.3f}, {m['CI_hi']:.3f}]  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}")
