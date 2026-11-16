"""Stage 6 classifier, validation and uncertainty utilities."""
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from . import config as C
from .data import impute_median


def _smote(X, y):
    try:
        k = min(C.SMOTE_K_MAX, int(y.sum()) - 1)
        return SMOTE(random_state=C.SEED, k_neighbors=k).fit_resample(X, y)
    except ValueError:        # too few minority cases in the fold
        return X, y


def loocv(df, feats, y=None, model='lr', keep_models=False):
    """Leave-one-subject-out CV. Scaling and SMOTE are fit on the training fold only.
    Returns probabilities, class predictions and (optionally) {index: (scaler, model)}."""
    X = impute_median(df, feats).values
    y = df['MCI'].values.astype(int) if y is None else np.asarray(y)
    prob, pred, fitted = np.zeros(len(y)), np.zeros(len(y)), {}
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr, ytr = _smote(sc.fit_transform(X[tr]), y[tr])
        clf = (LogisticRegression(**C.LR) if model == 'lr'
               else RandomForestClassifier(**C.RF_LOOCV)).fit(Xtr, ytr)
        Xte = sc.transform(X[te])
        prob[te], pred[te] = clf.predict_proba(Xte)[:, 1], clf.predict(Xte)
        if keep_models:
            fitted[te[0]] = (sc, clf)
    return (prob, pred, fitted) if keep_models else (prob, pred)


def boot_auc_ci(y, p, n=C.N_BOOT, seed=C.SEED):
    rng, b = np.random.RandomState(seed), []
    for _ in range(n):
        i = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(b, [2.5, 97.5]))


def metrics(y, p, pred, label='', ci=True):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    lo, hi = boot_auc_ci(y, p) if ci and not np.isnan(auc) else (np.nan, np.nan)
    div = lambda a, b: a / b if b else np.nan
    return dict(model=label, AUC=auc, CI_lo=lo, CI_hi=hi, accuracy=accuracy_score(y, pred),
                sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                PPV=div(tp, tp + fp), NPV=div(tn, tn + fn), F1=f1_score(y, pred, zero_division=0),
                Brier=brier_score_loss(y, p), TP=tp, FN=fn, FP=fp, TN=tn)


def delong(y, p1, p2):
    """Paired DeLong test for two correlated AUCs. Returns (AUC1 - AUC2, two-sided p)."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    def comps(p):
        v10 = np.array([np.mean((p[i] > p[neg]) + 0.5 * (p[i] == p[neg])) for i in pos])
        v01 = np.array([np.mean((p[pos] > p[j]) + 0.5 * (p[pos] == p[j])) for j in neg])
        return v10, v01
    a10, a01 = comps(p1); b10, b01 = comps(p2)
    S = np.cov(np.vstack([a10, b10])) / len(pos) + np.cov(np.vstack([a01, b01])) / len(neg)
    d = a10.mean() - b10.mean()
    return d, 2 * stats.norm.sf(abs(d / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])))


def paired_boot(y, p1, p2, n=C.N_BOOT, seed=C.SEED):
    rng, d = np.random.RandomState(seed), []
    for _ in range(n):
        i = resample(np.arange(len(y)), random_state=rng)
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return tuple(np.percentile(d, [2.5, 97.5])), 2 * min((d <= 0).mean(), (d >= 0).mean())


def permutation_test(df, feats, n=C.N_PERM, conditional_on=None, seed0=0):
    """Re-runs the full LOOCV pipeline under the null.
    conditional_on=None  -> shuffle labels (is the model better than chance?)
    conditional_on=[...] -> shuffle only the other features across participants,
                            keeping those columns and the labels fixed
                            (do the other features add anything?)."""
    y = df['MCI'].values.astype(int)
    obs = roc_auc_score(y, loocv(df, feats)[0])
    null = []
    for s in range(seed0, seed0 + n):
        rng = np.random.RandomState(s)
        if conditional_on is None:
            yp = rng.permutation(y)
            null.append(roc_auc_score(yp, loocv(df, feats, y=yp)[0]))
        else:
            d2 = df.copy(); other = [f for f in feats if f not in conditional_on]
            d2[other] = df[other].values[rng.permutation(len(df))]
            null.append(roc_auc_score(y, loocv(d2, feats)[0]))
    null = np.array(null)
    return dict(AUC=obs, p=(1 + (null >= obs).sum()) / (n + 1),
                null_mean=null.mean(), null_95=np.percentile(null, 95))


def calibration(y, p):
    """Logistic recalibration of the LOOCV predictions: logit(P(y=1)) = a + b * logit(p).
    Returns slope b, intercept a, and the Brier score after in-sample Platt scaling
    (illustrative only: the recalibration is fit on the same 51 predictions)."""
    import statsmodels.api as sm
    lg = np.log(p / (1 - p))
    a, b = sm.Logit(y, sm.add_constant(lg)).fit(disp=0).params
    platt = LogisticRegression(C=1e6).fit(lg[:, None], y).predict_proba(lg[:, None])[:, 1]
    return dict(slope=b, intercept=a, brier_raw=brier_score_loss(y, p),
                brier_platt_in_sample=brier_score_loss(y, platt),
                brier_no_skill=np.mean(y) * (1 - np.mean(y)))
