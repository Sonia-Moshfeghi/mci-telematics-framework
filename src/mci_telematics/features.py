"""Stage 1: variance inflation factor screening (single pass)."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from . import config as C
from .data import impute_median


def vif_table(df, feats):
    cols = [f for f in feats if f in df.columns and df[f].std() > 0]
    X = StandardScaler().fit_transform(impute_median(df, cols).values)
    v = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return pd.DataFrame({'feature': cols, 'VIF': v}).sort_values('VIF', ascending=False)


def vif_reduce(df, feats, threshold=C.VIF_THRESHOLD):
    """Drop every feature with VIF > threshold in ONE pass (order of `feats` preserved)."""
    t = vif_table(df, feats)
    keep = set(t.loc[t.VIF <= threshold, 'feature'])
    return [f for f in feats if f in keep]
