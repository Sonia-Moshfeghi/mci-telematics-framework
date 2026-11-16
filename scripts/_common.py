"""Shared setup for the analysis scripts: paths, data loading, output helpers."""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from mci_telematics import data   # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
FIGS = os.path.join(ROOT, 'results', 'figures')
os.makedirs(TABLES, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')


def args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'df.csv'),
                    help='trip-level CSV (see data/README.md)')
    if extra:
        extra(ap)
    return ap.parse_args()


def load(path, with_speed_cv=True):
    trips = data.load_trips(path)
    return trips, data.aggregate(trips, with_speed_cv=with_speed_cv)


def save(df, name, index=False):
    p = os.path.join(TABLES, name)
    df.round(4).to_csv(p, index=index)
    print(f'  -> results/tables/{name}')


def save_json(obj, name):
    with open(os.path.join(TABLES, name), 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'  -> results/tables/{name}')


def savefig(fig, name):
    fig.savefig(os.path.join(FIGS, name), dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'  -> results/figures/{name}')
