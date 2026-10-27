# Signal Hunters — Hack4Hack 2026 Round 2: Solution Report

## Headline (all numbers from executed runs; see `experiment_results_final.csv`)

| Validation | Macro-F1 (5-fold stratified, mean ± std) | Pooled OOF |
|---|---|---|
| Clean GBT held-out folds | **0.906 ± 0.006** (0.913 / 0.907 / 0.908 / 0.895 / 0.906) | 0.906 |
| Same folds under PKS-like perturbation (domain-shift stress test) | **0.812 ± 0.006** (0.820 / 0.813 / 0.812 / 0.803 / 0.814) | 0.812 |

The stretch target of 0.900 is met on clean GBT validation, but that is **not** a forecast of the PKS leaderboard. No GBT-only split reproduces the GBT→PKS shift exactly. The stress column is a proxy built from unlabeled PKS statistics. The real test may be easier or harder than this proxy, and its class mix is unknown. No leaderboard score has been observed.

## Data checks performed
- 18,000 × 6 × 16 × 128 uint8 train arrays and 10,000 test arrays; CSV rows align with array rows; sample-submission IDs equal the test IDs in the same order. No NaNs in the metadata.
- All 9 classes are present. The smallest is ETI with 479 examples (2.7%).
- `channel_width_hz` (2.79 vs 3.81) and `tsamp_s` (18.25 vs 16.0) are constant per telescope, so they only identify the telescope and are excluded. `center_freq_mhz` and `target` are excluded too. `slew_gap_rows` is used only inside the global-time track geometry, never as a model feature.
- GBT has a one-channel spike at channel 64 (≈130 vs ≈97), which is masked in both telescopes. PKS shows strong band-edge roll-off, ~2.6× more dropped rows, ~1.8× more glitch rows and larger panel gain spread.
- PKS log-noise is more negatively skewed with a lighter upper tail (z-quantiles 0.1%/99.9%: −4.09/2.87 vs GBT −3.79/3.73). This is consistent with fewer averaged degrees of freedom (log-Gamma, k≈4–5 vs ≈8). It explains the largest train–test feature shifts (KS up to 0.46 on pixel-maximum features; see `audit_shift.csv`).
- Drift-rate and dispersion-amplitude distributions in **pixel units** match between telescopes (ETI-like |drift| median 0.19 vs 0.16 ch/row; dispersion amplitude quantiles identical). The generator therefore works in pixel units, so no physical rescaling by tsamp or channel width is applied.

## Preprocessing (telescope-agnostic, per cadence, per panel)
0 → missing (NaN) and channel 64 → NaN. Broadband glitch rows are rejected when the row median is more than 0.6σ above the robust noise. The bandpass and panel gain are removed in the log domain (gain is additive there) using a 9-channel running median of the per-channel time-median, so narrow lines survive. Pixels are then converted to z-scores using the panel MAD. All statistics are missing-data aware.

## Physics-informed features (4 groups, 1,207 → top 350 used)
1. **Narrowband de-Doppler search.** Straight tracks are searched at slow (±1.5 ch/row, step 1/16), fast (±4) and steep (±16) drifts, scoring the sum of z divided by √(valid rows). Per panel this gives best significance, drift, line counts and steep-track counts. The **global ON/OFF track test** finds, for each drift d (step 0.01) and offset, the track channel in panel p at `c0 + d·p·(16+slew_gap_rows)`. Scores are min-ON, max-OFF, min-ON minus max-OFF and all-6 persistence, computed separately for drifting (|d| ≥ 0.045) and zero-drift tracks. **f⁻² dedispersion** uses f_lo/f_hi = 0.85, which the FRB sweep shapes fit. It covers 19 delay amplitudes on 32 sub-bands and gives pulse significance, pulse count (repetition, for pulsars) and ON/OFF counts. Pixel and texture statistics complete the group.
2. **Cross-panel (leave-one-panel-out) bandpass.** Group 1's per-panel smoothing absorbs broad structure. Here each panel is compared with the median spectrum of the other five, so broad ON-only brightening survives. Features are multi-scale smoothed texture, ON−OFF gain step and autocorrelations.
3. **Explicit ETI track consistency.** At the best global track, each ON panel is re-searched locally (±10 channels, all slow drifts). Features are the local position and drift offsets from the prediction, and the straight-line-fit residual of the three ON positions in global time. These defeat the "same drift but unconnected" trap: median offset is 1 channel and fit residual 0.33 for ETI, versus 8 and 3.3 for intermittent RFI. A 2-channel matched version is also included.
4. **Matched scintillation detector.** The 2D power spectrum of scintillating-star ON panels has 1.5–2.5× excess at k_t = 1–2, k_f = 0–4 (slowly varying broadband patches). The band power at those wavenumbers divided by high-k noise power is the best single scintillation feature (AUC 0.84), ranked 4th of all features.

The model learns the priority order from the labels. No hand-written rules override it.

## Domain-shift handling
- **PKS-like augmentation** (`augment.py`), with every parameter estimated from unlabeled test statistics:
  - Gamma(k ∈ [4, 8]) multiplicative noise in linear power.
  - Edge roll-off, tilt and ripple.
  - Panel gain jumps of σ = 2.5 units.
  - 0.9% glitch rows and 0.25% dropped rows.
  - The GBT spike is removed.
- Training uses the clean data plus one augmented copy. The validation-fold rows of every copy are excluded from training.
- **Stress validation** scores the held-out fold under the same perturbation, using a different random draw from the training copy.

## Model selection (fold 0, identical validation rows; full table in `experiment_results_final.csv`)
| Step | Clean | PKS-like |
|---|---|---|
| v1 features, clean training | 0.869 | 0.739 |
| + group 2 (cross-panel bandpass) | 0.907 | 0.783 |
| + group 3 + augmented training | 0.915 | 0.816 |
| + group 4, top-350 selection (fit 56 s vs ~600 s) | 0.913 | **0.820** |
| HGB 400 it / lr 0.06; 63 leaves | 0.911 / 0.916 | 0.818 / 0.815 (no gain) |
| ExtraTrees; HGB+ET blend | 0.893; ≤0.916 | 0.795; ≤0.820 (no gain) |
| Pseudo-labelling (transductive) | — | ≤0.822 (noise level, rejected) |
| Two augmented copies | 0.907 | 0.819 (rejected) |
| Per-class probability weights | — | no gain on held-out halves (rejected) |

**Chosen model:** HistGradientBoostingClassifier (balanced class weights, 150 iterations, learning rate 0.12, 64 bins). It uses the top 350 features by ExtraTrees importance, re-selected inside every CV fold so validation rows never influence selection, and averages 3 seeds at the final fit. It won because it is the simplest configuration that reached the best stress-test score. More capacity, ensembling, pseudo-labels and threshold tweaks gave no reproducible gain.

A CNN was not trained. The workspace had no GPU, 2 CPU cores, no PyTorch, and package installs were blocked. The physics features capture the joint six-panel structure explicitly.

## Per-class results (5-fold pooled OOF)
| Class | P clean | R clean | F1 clean | P PKS-like | R PKS-like | F1 PKS-like | n |
|:--|--:|--:|--:|--:|--:|--:|--:|
| eti_candidate | 0.838 | 0.802 | 0.820 | 0.778 | 0.557 | 0.650 | 479 |
| frb | 0.993 | 0.993 | 0.993 | 0.988 | 0.965 | 0.977 | 1042 |
| pulsar | 0.996 | 0.996 | 0.996 | 0.994 | 0.961 | 0.978 | 1629 |
| scintillating_star | 0.831 | 0.662 | 0.737 | 0.661 | 0.490 | 0.563 | 1431 |
| satellite_pass | 0.928 | 0.905 | 0.916 | 0.892 | 0.731 | 0.804 | 1805 |
| radar_chirp | 0.937 | 0.945 | 0.941 | 0.883 | 0.817 | 0.849 | 1849 |
| rfi_intermittent | 0.911 | 0.872 | 0.891 | 0.828 | 0.748 | 0.786 | 2662 |
| rfi_persistent | 0.864 | 0.963 | 0.910 | 0.792 | 0.908 | 0.846 | 2892 |
| noise | 0.933 | 0.961 | 0.947 | 0.786 | 0.951 | 0.861 | 4211 |

Confusion matrices are in `confusion_cleanval.csv` and `confusion_pksval.csv`, and example errors in `error_examples.png`. ETI false positives on clean data number 74 of 18,000: 66 are intermittent RFI, 6 persistent RFI and 1 each scintillating star and radar. ETI misses go mostly to intermittent RFI (57) and persistent RFI (22). These are faint ETI tracks sharing the window with bright RFI.

## Remaining weaknesses and risks
- **Scintillating stars** are the bottleneck (F1 0.74 clean / 0.56 stressed). Many are visually indistinguishable from noise at this SNR.
- **ETI recall drops under PKS-like noise** (0.80 → 0.56) because faint tracks fall below detection. Precision holds (0.78).
- **Signals are lost to `noise` under the stress test.** The model over-predicts `noise`: precision 0.79 stressed. If PKS signals are fainter than the proxy, this gets worse.
- **The augmentation is an approximation.** Its noise model is a fit to marginal pixel statistics, and the real PKS local-RFI frequencies and class mix are unknown.
- **Imperfect validation.** No GBT-only split can reproduce the GBT→PKS shift, and neither validation view is a guaranteed leaderboard proxy.

## Submission verification (`check_submission.py`)
All checks pass for `submission.csv`:
- exactly 10,000 rows;
- columns exactly `id,label`;
- IDs identical to `sample_submission.csv`, in the same order, and to `test_meta.csv`;
- IDs unique;
- no blank labels;
- all labels are among the 9 allowed strings.

Feature extraction reproduces the cache bit-exactly (`verify_features.py`, max |diff| = 0.0 on train, test and augmented slices).

## Rule compliance
Only the provided training data is used. Test cadences are used **unlabeled**, and only to estimate augmentation statistics and for the shift audit. There is no external data, no hand-labelling and no pseudo-labels in the final model. Seeds are fixed.
