"""Stage 3 (ANOVA + Tukey across clusters) and Stage 5 (Welch t-tests + BH-FDR)."""
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from . import config as C


def effect_label_eta(e):
    return 'Large' if e > 0.14 else 'Medium' if e > 0.06 else 'Small' if e > 0.01 else 'Negligible'


def effect_label_d(d):
    d = abs(d)
    return 'Large' if d >= 0.8 else 'Medium' if d >= 0.5 else 'Small' if d >= 0.2 else 'Negligible'


def anova_by_cluster(df, feats, cluster_col='cluster'):
    """Features where any cluster has < 2 valid observations are excluded (reported)."""
    rows, excluded = [], []
    ks = sorted(df[cluster_col].unique())
    for f in feats:
        groups = [df.loc[df[cluster_col] == k, f].dropna().values for k in ks]
        if not all(len(g) >= 2 for g in groups):
            excluded.append(f); continue
        F, p = f_oneway(*groups)
        m = df[f].mean()
        ss_b = sum(len(g) * (g.mean() - m) ** 2 for g in groups)
        ss_t = ((df[f] - m) ** 2).sum()
        eta = ss_b / ss_t if ss_t > 0 else 0
        rows.append(dict(feature=f, F=F, p=p, eta2=eta, effect=effect_label_eta(eta)))
    return pd.DataFrame(rows).sort_values('eta2', ascending=False), excluded


def tukey(df, feat, cluster_col='cluster'):
    r = pairwise_tukeyhsd(df[feat].values, df[cluster_col].values)
    return pd.DataFrame(r.summary().data[1:], columns=r.summary().data[0]).assign(feature=feat)


def welch_fdr(df, feats, q=C.FDR_Q):
    g1, g0 = df[df.MCI == 1], df[df.MCI == 0]
    rows = []
    for f in feats:
        a, b = g1[f].dropna(), g0[f].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = ttest_ind(a, b, equal_var=False)
        sp = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else 0
        rows.append(dict(feature=f, MCI_mean=a.mean(), nonMCI_mean=b.mean(), t=t, p=p,
                         cohen_d=d, effect=effect_label_d(d)))
    out = pd.DataFrame(rows)
    out['p_FDR'] = multipletests(out.p.values, method='fdr_bh')[1]
    out['FDR_sig'] = out.p_FDR < q
    return out.sort_values('cohen_d', key=abs, ascending=False)
