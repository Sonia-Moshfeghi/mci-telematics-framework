"""Fig. 7: raw LOOCV calibration and post-hoc (in-sample, illustrative) Platt recalibration."""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calibration_figure(y, p, cal, plt):
    lg = np.log(p / (1 - p))
    platt = LogisticRegression(C=1e6).fit(lg[:, None], y).predict_proba(lg[:, None])[:, 1]
    with plt.rc_context({'font.family': 'serif', 'axes.grid': False}):
        fig, axs = plt.subplots(1, 2, figsize=(9.9, 4.35))
        panels = [(p, '#C44E52', 'o', 'LR model (LOOCV)',
                   f"(a) Raw LOOCV calibration\nBrier={cal['brier_raw']:.3f}, "
                   f"slope={cal['slope']:.2f}, intercept={cal['intercept']:.2f}"),
                  (platt, '#55A868', 's', 'After Platt scaling',
                   f"(b) After post-hoc recalibration\nBrier={brier_score_loss(y, platt):.3f} "
                   "(illustrative; small-sample)")]
        for ax, (q, c, mk, lab, title) in zip(axs, panels):
            fr, mp = calibration_curve(y, q, n_bins=5, strategy='quantile')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
            ax.plot(mp, fr, marker=mk, color=c, lw=1.5, ms=7, label=lab)
            ax.plot(q, np.full_like(q, -0.03), '|', color='#4C72B0', alpha=0.5, ms=10)
            ax.set(xlim=(-0.02, 1.02), ylim=(-0.05, 1.05), xlabel='Mean predicted $P$(MCI)',
                   ylabel='Observed MCI fraction')
            ax.set_title(title, fontsize=9)
            ax.legend(loc='upper left', fontsize=8.5)
        fig.tight_layout()
    return fig, brier_score_loss(y, platt)
