"""Section 3.9: Random Forest ranking, SHAP, and exact per-participant
contributions from the leave-one-subject-out logistic regression (Fig. 14)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from . import config as C
from .data import impute_median


def rf_ranking(df, feats):
    """Random Forest fit once on all participants (descriptive ranking, not out-of-sample)."""
    X, y = impute_median(df, feats).values, df.MCI.values.astype(int)
    rf = RandomForestClassifier(**C.RF_RANKING).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    return rf, imp


def rf_shap(rf, df, feats):
    import shap
    X = impute_median(df, feats).values
    ex = shap.TreeExplainer(rf)
    sv = ex.shap_values(X)
    sv1 = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]
    return sv1, ex.expected_value[1], X


def lr_contributions(fitted_folds, df, feats, idx):
    """For a linear model, SHAP (interventional, independent features) equals
    coefficient x standardized value; computed with the fold that EXCLUDED `idx`."""
    sc, m = fitted_folds[idx]
    z = sc.transform(impute_median(df, feats).values[[idx]])[0]
    c = m.coef_[0] * z
    t = pd.DataFrame(dict(feature=feats, raw_value=df.loc[idx, feats].values, z=z,
                          coefficient=m.coef_[0], contribution_logit=c))
    return t.reindex(t.contribution_logit.abs().sort_values(ascending=False).index), m.intercept_[0]
