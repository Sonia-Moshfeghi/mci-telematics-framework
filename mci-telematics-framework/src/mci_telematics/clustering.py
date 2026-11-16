"""Stage 2 (K-Means profiling) and the mixed-type sensitivity analysis (Section 3.8.1)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from . import config as C
from .data import impute_median


def standardized(df, feats):
    """Primary encoding: ALL features z-scored, including the binary Gender/Retired."""
    return StandardScaler().fit_transform(impute_median(df, feats).values)


def explicit_encoding(df, feats):
    """Sensitivity encoding: continuous z-scored, binary kept as 0/1."""
    X = impute_median(df, feats)
    cont = [f for f in feats if f not in C.BINARY]
    return np.hstack([StandardScaler().fit_transform(X[cont].values), X[C.BINARY].values])


def elbow(Xs):
    return [KMeans(k, random_state=C.SEED, n_init=C.KMEANS_N_INIT_ELBOW).fit(Xs).inertia_
            for k in C.K_RANGE_ELBOW]


def kmeans(Xs, k=C.K):
    return KMeans(k, random_state=C.SEED, n_init=C.KMEANS_N_INIT).fit_predict(Xs)


def pca2(Xs):
    p = PCA(2, random_state=C.SEED).fit(Xs)
    return p.transform(Xs), p.explained_variance_ratio_


def chi2_vs_mci(labels, mci):
    """External validation only: MCI labels never enter the clustering."""
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(labels, mci))
    return chi2, p, dof


def mixed_type_sensitivity(df, feats, ks=range(2, 8)):
    import gower
    X = impute_median(df, feats)
    G = gower.gower_matrix(X.values, cat_features=[f in C.BINARY for f in feats])
    Xs, Xe = standardized(df, feats), explicit_encoding(df, feats)
    rows = []
    for k in ks:
        km_o = KMeans(k, random_state=C.SEED, n_init=C.KMEANS_N_INIT).fit_predict(Xs)
        km_e = KMeans(k, random_state=C.SEED, n_init=C.KMEANS_N_INIT).fit_predict(Xe)
        ag_c = AgglomerativeClustering(k, metric='precomputed', linkage='complete').fit_predict(G)
        ag_a = AgglomerativeClustering(k, metric='precomputed', linkage='average').fit_predict(G)
        rows.append(dict(K=k,
                         euclidean=silhouette_score(Xs, km_o),
                         explicit_encoding=silhouette_score(Xe, km_e),
                         gower_complete=silhouette_score(G, ag_c, metric='precomputed'),
                         gower_average=silhouette_score(G, ag_a, metric='precomputed')))
        if k == C.K:
            labels4 = dict(euclidean=km_o, explicit_encoding=km_e,
                           gower_complete=ag_c, gower_average=ag_a)
    sil = pd.DataFrame(rows)
    ext = pd.DataFrame([dict(method=m, **dict(zip(['chi2', 'p', 'dof'], chi2_vs_mci(l, df.MCI))))
                        for m, l in labels4.items()])
    return sil, ext, labels4
