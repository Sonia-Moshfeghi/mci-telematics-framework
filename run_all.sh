#!/usr/bin/env bash
# Reproduce every table and figure.   Usage: bash run_all.sh [path/to/df.csv] [n_permutations]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$(realpath "${1:-$ROOT/data/df.csv}")"
NPERM="${2:-1000}"
cd "$ROOT/scripts"
python 01_main_pipeline.py              --data "$DATA"
python 02_supplementary.py              --data "$DATA"
python 03_interpretability.py           --data "$DATA"
python 04_model_comparison.py --data "$DATA" --n-perm "$NPERM"   # ~10 min at 1000
python 05_sensitivity_analyses.py       --data "$DATA"
python 06_figures_10_11_12.py           --data "$DATA"
