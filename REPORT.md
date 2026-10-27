# Signal Hunters — Hack4Hack 2026 Round 2: Solution Report (final = v9)

All numbers below come from executed runs (`experiment_results.csv`). No leaderboard numbers are used except where the team reported them.

## Headline
**Final model v9** is the v8 recipe below with a larger booster: 300 iterations at learning rate 0.08 instead of 150 at 0.12. The CV table is for v8 (150 iterations, all 5 folds).

The larger booster was checked on 3 of the 5 folds, with identical folds and in-fold feature selection:

| Fold | Test-matched v8 → v9 | Clean v8 → v9 |
|---|---|---|
| 0 | 0.863 → 0.872 | 0.912 → 0.912 |
| 1 | 0.852 → 0.852 | 0.906 → 0.912 |
| 3 | 0.840 → 0.852 | 0.896 → 0.896 |

That is a mean of +0.007 on the test-matched proxy, with no fold worse. The other two folds were not run, for time.

| Validation (5-fold stratified, identical folds) | Macro-F1 mean ± std | Folds |
|---|---|---|
| Clean GBT | **0.903 ± 0.006** | 0.912 / 0.906 / 0.899 / 0.896 / 0.902 |
| **Test-matched PKS-like perturbation** (tracks the real LB) | **0.853 ± 0.008** | 0.863 / 0.852 / 0.861 / 0.840 / 0.851 |
| Harsh PKS-like perturbation (original stress test) | 0.810 ± 0.004 | 0.812 / 0.810 / 0.811 / 0.802 / 0.815 |

Real public-LB scores reported by the team: v2 = 0.8571, v3 = 0.8543. The test-matched proxy predicted ≈0.855 for that recipe, so it is a usable offline estimate of the leaderboard. The earlier harsh proxy under-estimated it by 0.04.

## What changed after the first leaderboard results (v3 → v8)
1. **Diagnosis: the augmentation was too strong.**
   - An adversarial classifier separated the augmented GBT copy from the real PKS test *more* easily (AUC 0.969) than clean GBT (0.904).
   - Pixel autocorrelation agrees (adjacent channel): PKS test 0.042, clean GBT 0.072, augmented 0.033.
   - The model had been trained for signals buried deeper than they are on PKS.
2. **Test-matched augmentation.** Gamma noise k ∈ [8,16] instead of [4,8], with the strength fitted to the test pixel autocorrelation (0.046 vs 0.042). Training on harsh + two test-matched copies scored best on every view. Fold-0 test-matched val:
   - none: 0.846
   - harsh: 0.857
   - matched: 0.855
   - harsh + matched: 0.861
   - **harsh + 2× matched: 0.863**
3. **Noise-calibrated detectors.**
   - Each panel is rescaled so its upper noise quantile matches across receivers.
   - Every drift, dedispersion and track statistic is expressed relative to its own panel's trial distribution: (S − median) / (q98 − median) × 2.054.
   - Pixel-extreme features (`zmax_`, `frac3_`, `frac5_`, `pos_mean_`) are dropped.
   - Measured on the real test data, this **halves the GBT-vs-PKS shift** of detector features (mean KS 0.081 → 0.042; dedispersion 0.189 → 0.018; ETI track 0.14 → 0.07).

## Pipeline
- **Preprocessing (`prep.py`):**
  - 0 → missing; GBT centre channel 64 masked.
  - Broadband glitch rows rejected.
  - Bandpass and gain removed in the log domain with a 9-channel running median, then MAD z-scores and noise-tail equalisation.
- **Features (1,190 → top 350 by ExtraTrees importance, selected inside each training fold):**
  1. **De-Doppler drift search** at slow, fast and steep drifts, plus the **global ON/OFF track test** in global time `p·(16+slew_gap_rows)+r`.
  2. **Cross-panel bandpass** (leave-one-panel-out), so broad ON-only emission survives.
  3. **ETI track-consistency refinement:** local position and drift offsets at each ON panel, and the line-fit residual of the three ON positions in global time. This defeats the "same drift, unconnected" trap.
  4. **Matched 2D power-spectrum scintillation detector** (AUC 0.84 alone; the existing features reach 0.95 scintillating-star-vs-noise AUC combined).
- **Model:** HistGradientBoosting with balanced class weights, 300 iterations, learning rate 0.08, 64 bins (v9; v8 used 150 at 0.12), averaged over 3 seeds. Trained on clean GBT + 1 harsh + 2 test-matched augmented copies (72,000 rows). No post-hoc thresholds: class-probability weights lowered held-out F1 by 0.003.
- **Metadata:** `channel_width_hz` and `tsamp_s` are constant per telescope (telescope identifiers), so they are excluded, as are `target` and `center_freq_mhz`. `slew_gap_rows` is used only in the track geometry.

## Per-class F1 (5-fold pooled OOF)
| Class | Clean | Test-matched | Harsh |
|---|---|---|---|
| eti_candidate | 0.816 | 0.728 | 0.659 |
| frb | 0.988 | 0.983 | 0.966 |
| pulsar | 0.996 | 0.989 | 0.968 |
| scintillating_star | 0.732 | 0.631 | 0.566 |
| satellite_pass | 0.908 | 0.839 | 0.787 |
| radar_chirp | 0.938 | 0.893 | 0.849 |
| rfi_intermittent | 0.893 | 0.843 | 0.789 |
| rfi_persistent | 0.912 | 0.877 | 0.845 |
| noise | 0.942 | 0.899 | 0.861 |

Confusion matrices: `confusion_v8_*.csv`.

## Tried and rejected (same validation rows)
| Idea | Result | Decision |
|---|---|---|
| More iterations / larger trees / ExtraTrees / HGB+ET blend (on 2-copy training data) | no gain | rejected; but with 4× training data, 300 iterations at lr 0.08 gave +0.007 → adopted in v9 |
| 500 instead of 350 features | fold 0: 0.865 vs 0.863 | not adopted |
| Pseudo-labelling (proxy) | ≤ +0.002 | not in final |
| Per-class probability weights | −0.003 on held-out halves | rejected |
| Per-line ON/OFF occupancy inventory (group 5) | 0.908 / 0.805 vs 0.909 / 0.808 | rejected (code kept) |
| Scintillation detectors in linear power | AUC 0.71 vs 0.84 (log) | rejected |
| Full low-k power spectrum for scintillation | AUC 0.956 vs 0.953 | negligible |
| **Numpy CNN** (no DL framework available; hand-written backprop, gradient-checked to 3e-8) | 0.60 clean after 5 epochs on 2 CPU cores | stopped: too slow to converge here |

## Limits and risks
- The weakest classes under PKS-like noise are scintillating stars (0.63), ETI (0.73) and satellites (0.84). Faint signals are lost to `noise`, so the score is sensitivity-limited rather than model-limited. A GPU-trained CNN is the most likely route to higher scores; it was not possible in this CPU-only, install-blocked environment.
- The augmentation matches pixel statistics, not the true PKS signal and class mix, so the test-matched proxy is an estimate, not a guarantee.

## Verification
- `check_submission.py`: 10,000 rows; columns exactly `id,label`; IDs identical to `sample_submission.csv` in the same order; unique; no blanks; all 9 labels valid. **ALL PASS.**
- `verify_features.py`: recomputed features equal the cache bit-exactly (max |diff| 0.0) for train, test and augmented slices.
- The final fit is deterministic (identical probabilities on rerun for the v3 recipe; v8 uses the same code path with fixed seeds).

## Rule compliance
- Only the provided training data is used, with no external data and no hand-labelling.
- Test cadences are used unlabeled, for augmentation and calibration statistics and for the shift audit.
- The final model uses no pseudo-labels.
