# Human-in-the-Loop Multi-Sensor Framework for MCI Driving Patterns: analysis code

Code for *"Human-in-the-Loop Multi-Sensor Framework for Identifying Driving Patterns Associated
with Mild Cognitive Impairment: A Proof-of-Concept Study"* (Moshfeghi et al.).

It reproduces every table and data figure in the manuscript (Figs. 3-14) from one trip-level CSV.

## Quick start

```bash
pip install -r requirements.txt
cp /path/to/df.csv data/df.csv          # participant data are not included
bash run_all.sh                          # ~12 min; outputs in results/tables and results/figures
```

Without the study data: `python tools/make_synthetic_data.py --out data/synthetic_df.csv`
then `bash run_all.sh data/synthetic_df.csv 20` (random values; demonstrates the code only).

## Layout

```
src/mci_telematics/config.py     every parameter, seed and threshold
src/mci_telematics/
  data.py          load trips, night/peak flags, participant-level aggregation
  features.py      Stage 1  VIF screening (single pass)
  clustering.py    Stage 2  K-Means, PCA, chi-square vs MCI; Gower/mixed-type sensitivity
  stats_tests.py   Stages 3 & 5  ANOVA + Tukey; Welch t-tests + Benjamini-Hochberg FDR
  models.py        Stage 6  LOOCV logistic regression / RF, metrics, bootstrap CIs,
                   calibration, DeLong, paired bootstrap, permutation tests
  interpret.py     RF ranking, SHAP, exact per-participant LR contributions
scripts/
  01_main_pipeline.py              Sections 3.1-3.7
  02_supplementary.py              Sections 3.8.1-3.8.5 and cold start (4.5)
  03_interpretability.py           Section 3.9 (Figs. 13-14)
  04_model_comparison.py           Section 3.8.2 (Table 7, DeLong and permutation tests)
  05_sensitivity_analyses.py       Appendix B (class-imbalance handling, within-fold imputation and VIF, air_sd)
  06_figures_10_11_12.py           Figs. 10, 11 and 12
tools/make_synthetic_data.py       schema-identical synthetic data
```

## Manuscript item → output

| Manuscript | Script | Output (`results/…`) |
|---|---|---|
| 3.1 Dataset characteristics | 01 | `tables/s3_1_dataset.json` |
| 3.2 VIF (11 retained features) | 01 | `tables/s3_2_vif_all_features.csv` |
| 3.3 Table 4, Fig. 3, χ² vs MCI | 01 | `tables/table4_cluster_centroids.csv`, `tables/s3_3_cluster_vs_mci_chi2.json`, `figures/fig3_kmeans.png` |
| 3.4 ANOVA, Tukey, Figs. 4-5 | 01 | `tables/s3_4_anova*.csv/json`, `tables/s3_4_tukey_top3.csv` |
| 3.5 RF importance, Fig. 6, RF LOOCV | 01 | `tables/fig6_rf_importance.csv`, `tables/s3_5_*` |
| 3.6 Table 5 | 01 | `tables/table5_welch_top10.csv` (all 36: `table5_welch_all.csv`) |
| 3.7 Table 6, Figs. 7-8, coefficients, calibration | 01 | `tables/table6_lr_loocv.csv`, `tables/s3_7_*`, `figures/fig7_calibration.pdf`, `figures/fig8_lr_results.png` |
| 3.8.1 Fig. 9 (clustering sensitivity) | 02 | `tables/s3_8_1_*` |
| 3.8.2 Table 7, DeLong and permutation tests | 04 | `tables/table7_model_comparison.csv`, `tables/s3_8_2_*` |
| 3.8.3 Table 8 | 02 | `tables/table8_incremental.csv` |
| 3.8.4 Fig. 10 (temporal split) | 02 | `tables/s3_8_4_*` |
| 4.5 Cold start | 02 | `tables/s4_5_cold_start.csv` |
| 3.8.5 Table 9, Figs. 11-12 | 02 | `tables/table9_subgroups.csv`, `tables/s3_8_5_equalized_odds.csv` |
| Appendix B, Tables B1 and B2 | 05 | `tables/tableB1_main_model_sensitivity.csv`, `tables/tableB2_class_imbalance.csv`, `tables/sA_*.json` |
| Figs. 10, 11 and 12 | 06 | `figures/fig10_temporal_stability.pdf`, `figures/fig11_subgroups.pdf`, `figures/fig12_equalized_odds.pdf` |
| 3.9 Fig. 13 (RF SHAP) | 03 | `figures/fig13_rf_shap_beeswarm.png` |
| 3.9 Fig. 14 | 03 | `figures/fig14_lr_contributions.pdf` |

## Verification

Run on the study data, the code reproduces the manuscript: 51 participants / 10 MCI /
20,145 trips; 11 VIF-retained features; cluster sizes 19/2/3/27, χ²(3) = 1.78, p = 0.619;
hturn_value η² = 0.964; thro_sd RF importance 0.083; Welch thro_sd d = 0.86, Gender
p_FDR = 0.008; LR AUC 0.698 [0.493, 0.872], sensitivity 0.800, specificity 0.659,
Brier 0.224, calibration slope 0.77 / intercept −1.48, Platt-recalibrated Brier 0.141; Table 7-9, cold-start
(0.776 at 50 trips), temporal (AUC 1.000, n = 14) and equalized-odds values as reported.

Model comparisons (1,000 permutations, seeds 0-999): Age + Gender AUC 0.763,
p = 0.010; full model 0.698, p = 0.070; telematics only 0.595, p = 0.229; full without
Gender 0.598, p = 0.232; demographics only 0.722, p = 0.039; telematics block added to
Age + Gender, conditional p = 0.319.

Sensitivity analyses of the main model: class weights only 0.707 [0.507, 0.882]; SMOTE-NC
0.722 [0.519, 0.894]; median imputation within folds 0.698 [0.493, 0.872]; imputation and VIF
screen within folds 0.705 [0.502, 0.869]; excluding air_sd 0.717 [0.513, 0.889].

## Data protection

The study data cannot be shared and are not part of this package. Running the code
writes participant-level outputs to `results/` (e.g. `loocv_predictions_full_model.csv`,
`fig14_contributions_*.csv`, `s3_8_4_temporal_predictions.csv`); `results/` and
`data/*.csv` must not be redistributed.
