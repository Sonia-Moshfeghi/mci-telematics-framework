"""Section 3.8.2: does multi-sensor driving data add predictive value over demographics?
Table 7 (all six model variants), paired DeLong and bootstrap comparisons, and
whole-pipeline permutation tests.

Permutation tests take ~2 min per model at 1,000 permutations; use --n-perm to change."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from _common import args, load, save, save_json
from mci_telematics import config as C, features, models

a = args(lambda ap: ap.add_argument('--n-perm', type=int, default=C.N_PERM))
trips, dfp = load(a.data, with_speed_cv=False)
y = dfp.MCI.values.astype(int)
TEL = C.EXPOSURE + C.KINEMATIC + C.VEHICLE
S = {'Age + Gender': ['Age', 'Gender'],
     'Demographics only': features.vif_reduce(dfp, C.DEMO),
     'Telematics only': features.vif_reduce(dfp, TEL),
     'Full (Demo + Telematics)': features.vif_reduce(dfp, C.DEMO + TEL)}
S['Full without Gender'] = [f for f in S['Full (Demo + Telematics)'] if f != 'Gender']
S['Age + Gender + Telematics'] = features.vif_reduce(dfp, ['Age', 'Gender'] + S['Telematics only'])
save_json(S, 'table7_model_comparison_feature_sets.json')

print('Table 7')
P, rows = {}, []
for k, f in S.items():
    p, d = models.loocv(dfp, f); P[k] = p
    rows.append(dict(models.metrics(y, p, d, k), n_features=len(f)))
t7 = pd.DataFrame(rows)

print('Paired comparisons')
pairs = [('Full (Demo + Telematics)', 'Demographics only'), ('Full (Demo + Telematics)', 'Age + Gender'),
         ('Age + Gender + Telematics', 'Age + Gender'), ('Telematics only', 'Demographics only'),
         ('Full (Demo + Telematics)', 'Full without Gender')]
cmp = []
for x, z in pairs:
    dd, p_dl = models.delong(y, P[x], P[z]); (lo, hi), p_b = models.paired_boot(y, P[x], P[z])
    cmp.append(dict(model=x, baseline=z, delta_AUC=dd, DeLong_p=p_dl, boot_CI_lo=lo, boot_CI_hi=hi, boot_p=p_b))
save(pd.DataFrame(cmp), 's3_8_2_paired_auc_comparisons.csv')

print(f'Permutation tests ({a.n_perm} permutations each)')
perm = {}
for k in ['Age + Gender', 'Demographics only', 'Telematics only', 'Full without Gender', 'Full (Demo + Telematics)']:
    perm[k] = models.permutation_test(dfp, S[k], n=a.n_perm)
    print(f"  label-permutation  {k:28s} AUC={perm[k]['AUC']:.3f} p={perm[k]['p']:.3f}")
cond = models.permutation_test(dfp, S['Age + Gender + Telematics'], n=a.n_perm, conditional_on=['Age', 'Gender'])
print(f"  telematics-block   Age + Gender + Telematics    AUC={cond['AUC']:.3f} p={cond['p']:.3f}")
t7['permutation_p'] = t7.model.map({k: v['p'] for k, v in perm.items()})
t7.loc[t7.model == 'Age + Gender + Telematics', 'permutation_p'] = cond['p']
t7['permutation_test'] = t7.model.map(lambda m: 'telematics block shuffled (conditional)'
                                      if m == 'Age + Gender + Telematics' else
                                      ('labels shuffled' if m in perm else ''))
save(t7, 'table7_model_comparison.csv')
save_json(dict(label_permutation=perm, conditional_telematics=cond, n_perm=a.n_perm), 's3_8_2_permutation_tests.json')
