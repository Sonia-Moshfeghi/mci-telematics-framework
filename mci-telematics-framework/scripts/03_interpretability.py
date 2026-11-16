"""Section 3.9 interpretability.
 * Fig. 13: SHAP summary for the Random Forest ranking model (fit on all 51 participants;
   descriptive, not out-of-sample).
 * Fig. 14: exact per-participant contributions from the leave-one-subject-out
   logistic regression, i.e. the classifier whose predictions are reported, computed with
   the fold that excluded the participant shown."""
import numpy as np
import pandas as pd
from _common import args, load, save, savefig, plt
from mci_telematics import config as C, features, interpret, models

a = args()
trips, dfp = load(a.data)
y = dfp.MCI.values.astype(int)
F = C.MAIN_FEATURES
REDUCED = features.vif_reduce(dfp, F)

print('Fig. 13 - Random Forest SHAP (global)')
rf, imp = interpret.rf_ranking(dfp, F)
sv, base, X = interpret.rf_shap(rf, dfp, F)
save(pd.DataFrame(dict(feature=F, mean_abs_shap=np.abs(sv).mean(0))).sort_values('mean_abs_shap', ascending=False),
     'fig13_rf_mean_abs_shap.csv')
import shap
shap.summary_plot(sv, X, feature_names=F, max_display=15, show=False)
savefig(plt.gcf(), 'fig13_rf_shap_beeswarm.png')

print('Fig. 14 - LOSO logistic-regression contributions')
p, d, folds = models.loocv(dfp, REDUCED, keep_models=True)
tp_idx = int(np.argmax(np.where(y == 1, p, -1)))                       # most confidently flagged MCI case
miss = np.where((y == 1) & (dfp.Gender.values == 1))[0]
miss_idx = int(miss[0]) if len(miss) else int(np.argmin(np.where(y == 1, p, 2)))
labels = {'Age': 'Age', 'Gender': 'Gender (male=1)', 'Education': 'Education', 'BMI': 'BMI category',
          'Retired': 'Retired', 'nNightTrip': 'Night trips', 'n_ACC': 'Hard accel. (n/trip)',
          'n_HTURN': 'Hard turns (n/trip)', 'hturn_value': 'Turn severity', 'air_sd': 'Ambient temp. SD',
          'thro_mean': 'Throttle mean'}
fig, axs = plt.subplots(2, 1, figsize=(6.2, 7.2))
for ax, i, title in [(axs[0], tp_idx, '(a) Correctly flagged MCI participant'),
                     (axs[1], miss_idx, '(b) Missed male MCI participant')]:
    t, intercept = interpret.lr_contributions(folds, dfp, REDUCED, i)
    save(t.assign(participant_id=dfp.participant_id[i], loso_prob=p[i], intercept=intercept),
         f'fig14_contributions_{"a" if i == tp_idx else "b"}.csv')
    t = t.iloc[::-1]
    ax.barh([labels.get(f, f) for f in t.feature], t.contribution_logit,
            color=['#C44E52' if v > 0 else '#4C72B0' for v in t.contribution_logit])
    ax.axvline(0, c='k', lw=.8); ax.set_xlabel('Contribution to log-odds of MCI (coefficient \u00d7 standardized value)', fontsize=9); ax.tick_params(labelsize=9)
    ax.set_title(f'{title}\nheld-out P(MCI) = {p[i]:.2f}  (threshold {C.THRESHOLD:.2f})', fontsize=10)
fig.tight_layout(); fig.savefig(__import__('os').path.join(__import__('_common').FIGS, 'fig14_lr_contributions.pdf'), bbox_inches='tight'); savefig(fig, 'fig14_lr_contributions.png')
