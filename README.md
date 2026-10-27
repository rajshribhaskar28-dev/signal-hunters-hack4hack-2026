# Signal Hunters — Hack4Hack 2026 Round 2 solution

Physics-informed, noise-calibrated feature extraction (de-Doppler drift search, global ON/OFF
track geometry, f⁻² dedispersion, cross-panel bandpass, ETI track consistency, matched
scintillation detector) + HistGradientBoosting trained on clean GBT data plus three PKS-like
augmented copies (one harsh, two matched to the real test noise). Details: **REPORT.md**.

5-fold CV macro-F1: **0.903 ± 0.006** (clean GBT), **0.853 ± 0.008** (test-matched PKS-like
proxy, which tracked the real public LB), 0.810 ± 0.004 (harsh proxy) for v8. The final v9 uses a larger booster (300 iterations, learning rate 0.08): about +0.007 on the test-matched proxy over 3 folds checked (see REPORT.md).

## Reproduce the submission
```bash
pip install -r requirements.txt
# put train_cadences.npy, test_cadences.npy, train_meta.csv, test_meta.csv, sample_submission.csv in DATA_DIR
python train_pipeline.py --data DATA_DIR          # features (~2 h on 2 CPU cores) + fit (~15 min)
python train_pipeline.py --data DATA_DIR --cv     # also runs the fold-0 clean / PKS-like validation
SH_DATA=DATA_DIR python check_submission.py submission.csv   # schema checks
```
Feature caches are written to `./cache`; rerunning skips groups already extracted.
CPU only, ~2 GB RAM peak. Seeds fixed (CV split seed 0; augmentation copies 2 (harsh), 4 and 5 (test-matched);
HGB seeds 0,1,2 averaged).

## Files
| File | Purpose |
|---|---|
| `prep.py` | missing-data/spike/glitch masking, bandpass + gain flattening, robust z-scores |
| `drift.py` | vectorised straight-track drift sums |
| `features.py` | group 1: drift search, global ON/OFF track test, dedispersion, texture |
| `features2.py` | group 2: leave-one-panel-out bandpass, broad ON-only emission |
| `features3.py` | group 3: ETI track-consistency refinement |
| `features4.py` | group 4: line-robust texture, 2-D power-spectrum scintillation detector |
| `features5.py` | group 5 (evaluated, not used): per-line panel-occupancy inventory |
| `cnn.py`, `train_cnn.py`, `prep_cnn.py`, `blend_eval.py` | numpy CNN experiment (evaluated, not used) |
| `augment.py` | PKS-like perturbation (parameters from unlabeled test statistics) |
| `train.py`, `final_model.py`, `final_config.json` | model definition, CV, feature selection, final fit |
| `train_pipeline.py` | end-to-end entry point |
| `audit.py`, `cv_summary.py`, `tune_weights.py`, `exp*.py`, `select_feats.py` | experiments / audits used during development |
| `verify_features.py`, `check_submission.py` | reproducibility and schema checks |
| `experiment_results_final.csv` | every measured experiment with conclusions |
| `audit_shift.csv` | GBT vs PKS feature-distribution audit (KS statistics) |
| `per_class_v8_*.csv`, `confusion_v8_*.csv`, `error_examples.png` | final CV diagnostics |

## Submission checklist
1. **Leaderboard tab (Round 2)** → upload `submission.csv`. Max 5 uploads; files that fail format
   checks are rejected without using an upload. The final ranking uses your best public upload
   scored on the hidden 70%.
2. **Submission tab** → paste the public GitHub repository (or notebook) link containing this folder
   **before 11:59 PM IST, 27 Oct 2026**.
3. Optional: confirm the repo is public and that `python train_pipeline.py --data ...` is documented
   (it is, above).
