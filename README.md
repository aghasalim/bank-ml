# bank-ml: leakage-aware classification on UCI Bank Marketing

**Problem.** Predict whether a bank client subscribes to a term deposit after a marketing call (45,211 rows, 11.7% positive; UCI Bank Marketing `bank-full.csv`, CC BY 4.0).
**What this repo is about:** doing the evaluation honestly. Two things make the naive numbers misleading, and both are measured below.

1. **Target leakage.** `duration` (call length) is only known after the call outcome. It is excluded from the main models, and kept in one ablation to show how much it inflates scores.
2. **Time drift.** Rows are in chronological order (UCI documentation), and the positive rate moves from about 3% in the earliest eighth of the rows to about 45% in the last eighth. Train (first 80%) has 6.7% positives; the chronological test set (last 20%) has **31.6%**. A random split hides this.

## Setup
Split: first 80% of rows = train, last 20% = test (n_train=36,168, n_test=9,043). All preprocessing (one-hot with rare-category grouping, scaling for logistic regression) is inside an sklearn `Pipeline`, so no fold sees another fold's statistics. Tuning touched only the train portion. The test set was evaluated once per model.
Metric choice: because prevalence differs between folds and test, PR-AUC is not comparable across them (a no-skill model scores 0.316 on test). ROC-AUC is prevalence-insensitive, so it is the headline; PR-AUC is reported as lift over prevalence.

## Results (all rows run; defaults except where noted)
| model | duration | random-CV ROC-AUC | time-CV ROC-AUC | **test ROC-AUC** [95% CI] | test PR-AUC lift | Brier | ECE |
|---|---|---|---|---|---|---|---|
| dummy | dropped | 0.500±0.000 | 0.500±0.000 | 0.5 (0.5, 0.5) | 1.0 | 0.278 | 0.249 |
| logreg | dropped | 0.682±0.007 | 0.561±0.057 | 0.683 (0.671, 0.696) | 1.54 | 0.231 | 0.184 |
| rf | dropped | 0.704±0.008 | 0.543±0.027 | 0.696 (0.685, 0.708) | 1.5 | 0.225 | 0.163 |
| xgb | dropped | 0.716±0.008 | 0.546±0.031 | 0.69 (0.679, 0.703) | 1.56 | 0.227 | 0.167 |
| lgbm | dropped | 0.712±0.010 | 0.542±0.042 | 0.682 (0.67, 0.694) | 1.52 | 0.228 | 0.169 |
| logreg | used | 0.933±0.002 | 0.881±0.071 | 0.765 (0.755, 0.775) | 1.77 | 0.218 | 0.187 |
| rf | used | 0.948±0.003 | 0.862±0.102 | 0.776 (0.767, 0.785) | 1.78 | 0.21 | 0.167 |
| xgb | used | 0.954±0.003 | 0.886±0.078 | 0.79 (0.781, 0.8) | 1.82 | 0.221 | 0.187 |
| lgbm | used | 0.955±0.003 | 0.887±0.077 | 0.786 (0.777, 0.796) | 1.81 | 0.223 | 0.189 |

Reading it:
- **Leakage:** with `duration`, LightGBM scores ROC-AUC 0.955 under random CV but 0.786 on the chronological test. Without it: 0.712 vs 0.682. A random-CV-only report would have claimed ~0.95.
- **Model choice barely matters.** Without `duration`, all four models land at 0.68–0.70 test ROC-AUC with overlapping CIs. Gradient boosting wins random CV by about 0.03 over logistic regression, but that edge does not carry over to the later period.
- **Time-CV on train is low and noisy** (about 0.55 ± 0.03–0.06) because the first expanding-window folds train on a few thousand rows from a 3% prevalence period.

## Tuning (LightGBM, no duration, 25 Optuna trials, expanding-window CV on train only)
```
best params: {'n_estimators': 188, 'learning_rate': 0.02684026549963137, 'num_leaves': 7, 'min_child_samples': 100, 'reg_lambda': 0.42611740930372816}
time-CV ROC tuned=0.567
test ROC default=0.682 (np.float64(0.67), np.float64(0.694))  tuned=0.705 (np.float64(0.693), np.float64(0.715))
```
Test ROC-AUC rose from 0.682 to 0.705. The bootstrap CIs barely overlap, so this is suggestive, not conclusive (no paired test was run). The tuned model is smaller and more regularised (7 leaves).

## Calibration (fit on first 80% of train, calibrate on the most recent 20% of train, evaluate on test)
```
calibration slice prevalence=0.137, test prevalence=0.316
uncalibrated  brier=0.283 ece=0.267 mean_pred=0.049
sigmoid       brier=0.244 ece=0.186 mean_pred=0.130
isotonic      brier=0.243 ece=0.185 mean_pred=0.131

```
Calibration helps (ECE 0.267 -> about 0.185) but the model stays badly calibrated: it was trained at 6.7% positives and the test period has 31.6%. This is prior shift, and recalibrating on the most recent training slice (13.7% positives) cannot fix a shift that large. Plot: `results/calibration.png`. In production this model would need recalibration on fresh labels.

## Error analysis by segment (tuned model, test, n >= 150; `results/segments.csv`)
| segment          |    n |   prevalence |   mean_pred |   roc_auc |
|:-----------------|-----:|-------------:|------------:|----------:|
| poutcome=success | 1301 |        0.695 |       0.271 |     0.545 |
| poutcome=failure | 2224 |        0.214 |       0.079 |     0.599 |
| poutcome=other   |  909 |        0.271 |       0.092 |     0.627 |
| age_band=60+     | 1094 |        0.435 |       0.204 |     0.632 |
| poutcome=unknown | 4609 |        0.267 |       0.098 |     0.638 |
| job=retired      |  897 |        0.419 |       0.196 |     0.644 |
| segment           |   n |   prevalence |   mean_pred |   roc_auc |
|:------------------|----:|-------------:|------------:|----------:|
| contact=unknown   | 255 |        0.129 |       0.08  |     0.732 |
| contact=telephone | 899 |        0.263 |       0.12  |     0.749 |
| job=self-employed | 274 |        0.321 |       0.127 |     0.782 |

The model is weakest for clients with a previous successful campaign (`poutcome=success`: ROC-AUC 0.545; 69.5% actually subscribed, mean prediction 27%) and for older/retired clients. It underpredicts exactly the segment that converts most.

## SHAP (tuned model, 2000 test rows; `results/shap_summary.png`)
|                  |      0 |
|:-----------------|-------:|
| poutcome_success | 0.1784 |
| age              | 0.1444 |
| balance          | 0.1289 |
| contact_unknown  | 0.1181 |
| marital_married  | 0.113  |
| day              | 0.0931 |
| pdays            | 0.0914 |
| month_apr        | 0.077  |
| month_feb        | 0.065  |
| month_mar        | 0.0635 |

`day` and `month_*` features rank high. They are partly proxies for *when* a campaign ran, so they encode the drift, not client behaviour. I did not remove them; treat their importance with caution.

## Run it
```bash
make setup data test
make baselines   # a few minutes
make final       # tuning + calibration + segments + SHAP
```
`docker build -t bank-ml .` runs the tests (**Docker build not tested**).

## Limitations
- One chronological split; no repeated-split uncertainty over time. CIs only reflect bootstrap resampling of the test rows.
- The year is not in the data, so the time ordering relies on the documented row order and the month sequence.
- ROC-AUC and PR-lift only; no cost-sensitive threshold analysis.
- 25 tuning trials on one model; the other models use fixed defaults, so the model comparison favours the tuned LightGBM.
- Single dataset, from one Portuguese bank, 2008–2010.
