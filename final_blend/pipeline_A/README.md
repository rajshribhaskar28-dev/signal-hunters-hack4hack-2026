# Signal Hunters — Hack4Hack 2026 Round 2

Physics-informed features + PKS-like domain augmentation + a HistGradientBoosting ensemble.
Uses only the provided competition data and open-source libraries (numpy, scipy, pandas, scikit-learn).

## Reproduce

```bash
pip install -r requirements.txt
python train_pipeline.py --data /path/to/signal-hunters-round2 --out submission.csv          # ~60-70 min on 2 CPU cores
python train_pipeline.py --data /path/to/signal-hunters-round2 --out submission.csv --cv     # also re-runs 5-fold CV
```

The first run extracts and caches features in `./cache`; later runs reuse them. All seeds are fixed.

## Results (every number comes from an actual run; see `experiment_results.csv`)

Validation:
- **Clean:** 5-fold stratified CV on GBT train (seed 42).
- **Stress test:** the same folds, but each held-out fold is passed through the PKS-like augmentation. This simulates the GBT→PKS shift.

| Exp | Change | Macro-F1, clean CV | Macro-F1, PKS stress |
|---|---|---|---|
| E0 | starter-notebook baseline | 0.541 | – |
| E1 | physics features v1 (drift search, global ETI track, de-dispersion, texture) | 0.842 | – |
| E2 | + ON/OFF level & patchiness (scintillation), asymmetric OFF veto | 0.875 | – |
| E3 | + 1/120 ch/row global drift grid, line maps on z clipped at 6σ | 0.880 | 0.605 (trained on clean only) |
| E4 | + training on clean + PKS-augmented copies | 0.883 | 0.825 |
| E5/E6 | + track occupancy; drop telescope-dependent noise-floor stats | 0.882 | 0.827 |
| E7 | 2-feature-set HGB ensemble | 0.883 | 0.829 |
| **E8 (final)** | **+ ON-track collinearity / drift-consistency geometry** | **0.898 ± 0.005** | **0.842 ± 0.005** |

(± = std of macro-F1 across the 5 folds.)

Per-class results, final config, OOF:

| Class | P (clean) | R (clean) | F1 (clean) | F1 (PKS stress) |
|---|---|---|---|---|
| eti_candidate | 0.889 | 0.756 | 0.817 | 0.686 |
| frb | 0.979 | 0.977 | 0.978 | 0.957 |
| pulsar | 0.995 | 0.985 | 0.990 | 0.977 |
| scintillating_star | 0.832 | 0.652 | 0.731 | 0.619 |
| satellite_pass | 0.932 | 0.894 | 0.913 | 0.871 |
| radar_chirp | 0.932 | 0.920 | 0.926 | 0.881 |
| rfi_intermittent | 0.908 | 0.858 | 0.882 | 0.823 |
| rfi_persistent | 0.853 | 0.957 | 0.902 | 0.868 |
| noise | 0.914 | 0.976 | 0.944 | 0.895 |

Confusion matrices: `confusion_final.png`, `final_confusion_c.csv` (clean) and `final_confusion_a.csv` (stress).

## Method

**Preprocessing (`prep.py`)**
- Value 0 = dropped row. It is masked throughout and never treated as low power.
- The GBT centre-channel spike (ch 64) is replaced by the mean of ch 63/65, identically in train and test.
- Broadband glitch rows (|row median − panel median| > 0.35 log2) are masked.
- Linear power is divided by a per-panel bandpass: the per-channel median over time, then a 9-channel running median, so narrow lines survive.
- Robust MAD noise normalisation gives a standardized excess-power map z.

**Features (`features.py`, `geom.py`)**, ~785 features. Every feature group is computed per panel, then summarised as ON/OFF aggregates (mean, min, max, ON−OFF).

- **A. Narrowband drift search.** A sparse-matrix line sum over 91 drift rates (±1.5 ch/row) × 128 start channels per panel. It reports line SNR, drift, zero-drift vs drifting lines, and line counts.
- **B. Global ON/OFF track consistency.**
  - Global time is `p·(16+slew_gap_rows)+r`. A 361-step global drift grid (1/120 ch/row) predicts each panel's channel.
  - ON evidence uses tight ±1-channel geometry. The OFF veto uses a deliberately loose ±4-channel, ±1-drift window.
  - Line maps for this search use z clipped at 6σ, so bright RFI crossing a track can't fake support.
  - Scores: min-ON, ON−OFF, a sum score, and a zero-drift-only variant (the "ON-only but not drifting" trap).
  - **Geometry refinement (v5):** sub-channel line positions in each ON panel. Collinearity residual `x₃ − (x₁+x₅)/2` (exact, because panel 3's mid-time is the midpoint of panels 1 and 5). Also the mismatch between in-panel drift and global drift.
  - This exposed the "same drift but doesn't connect" trap: traps are offset by 1–3 channels (median residual 1.4 ch), true ETIs by 0.08 ch. ETI F1 went 0.71 → 0.82.
- **C. RFI persistence.** Occupancy of the brightest global track: panels present vs panels in which the track is in-band. Persistent lines drift continuously and "fade" when they leave the band.
- **D/E. Satellites and radar.** Steep drift search (1–10 ch/row) with per-panel maxima, panel adjacency and runs. Repeated parallel tracks and 2-D FFT lattice peaks for sawtooth radar.
- **F. Pulsars and FRBs.** De-dispersion with an f⁻²-shaped delay curve swept over 18 delay spans (2–104 rows). The curve shape (f₀≈400 in channel units) was **fitted empirically on training sweeps**, because the metadata's physical band (128 × 2.79 Hz) implies zero dispersion, so physical DM units don't apply. Also: per-panel max SNR, a repeat count (pulsar) and a reversed-sweep control.
- **G. Scintillation.** Panel power level relative to the cadence (scintillating stars raise ON power by ~12%). An alternating ON/OFF level pattern against gain wander. Gain-robust spectral patchiness relative to OFF panels. Row-level variability. ON block excess over OFF.
- **H. Instrument.** Dropped/glitch row counts and texture statistics.

**Domain shift (`augment.py`)**
- `tsamp_s` and `channel_width_hz` are constant per telescope (pure telescope ID). `center_freq_mhz` has no label dependence and a different range on PKS. `target` is declared useless. All four are excluded; only `slew_gap_rows` is used.
- The PKS-like augmentation is fitted **only to unlabeled test statistics**:
  - bandpass difference profile with random tilt and scale;
  - per-panel gain jitter (within-cadence level std 1.75 → 2.6);
  - left-skewed log-domain noise (Gamma k=0.5, 11.5 value units), chosen by grid search to match test residual quantiles and z-tail statistics within ~3%;
  - extra glitch rows (0.8% → 1.7%) and dropped rows (0.15% → 0.4%).
- Audit: median per-feature KS distance to test falls from 0.098 (clean GBT) to 0.033 (augmented). Adversarial train-vs-test AUC falls from 1.000 to 0.864; some separability is expected because the class mix differs.
- A model trained on clean GBT only collapsed on the stress holdout (macro 0.605, noise F1 0.04). The same model trained with augmentation keeps 0.84 there and loses nothing on clean data.
- 130 absolute noise-floor statistics flagged by the audit were dropped from one ensemble member (no CV cost). Their ON−OFF contrasts were kept.

**Model**
- `HistGradientBoostingClassifier` (lr 0.1, 31 leaves, L2=1, balanced class weights, early stopping).
- Two feature subsets × 3 seeds, probability-averaged, trained on clean train + 2 augmented copies (54,000 rows).

**Tried and rejected (no reproducible gain)**
- Per-class probability offsets: ±0.001 on an honest half-split.
- ETI probability thresholds: flat optimum, so I keep argmax.
- ExtraTrees: too slow on 2 CPUs within the time limit.
- No CNN: PyTorch couldn't be installed in the competition environment.

## Limitations and risks

- No GBT-only split truly reproduces GBT→PKS. The stress test uses a *simulated* shift fitted to unlabeled test statistics, so the real PKS score could differ either way.
- Time resolution (16 s vs 18.25 s) and channel width differ, so physical drift rates could have been rescaled in channels/row. I checked this on unlabeled data: drift and sweep-duration distributions of strong signals match between train and test (e.g. steep-line drift quartiles 1.0/1.5/6.5 ch/row in both). No rescaling is evident.
- The test label mix is unknown and differs from train. Balanced class weights reduce prior dependence, but no prior correction was applied.
- `scintillating_star` (0.73 / 0.62) and faint `eti_candidate` remain SNR-limited. ETI recall is 0.76 on clean CV; misses are concentrated at summed track SNR < 7.

## Final submission: `submission_final.csv`

- **Model:** 6 HistGradientBoosting models (2 feature subsets × 3 seeds), probability-averaged. Trained on 18,000 clean GBT cadences + 2 × 18,000 PKS-augmented copies. Training takes ~21 min on 2 CPUs.
- **Schema checks** (all PASS):
  - columns exactly `id,label`;
  - 10,000 rows;
  - IDs unique and identical to `sample_submission.csv` order;
  - no blank labels;
  - every label is one of the 9 permitted strings.
- **Predicted test mix (%):** noise 22.8, rfi_intermittent 17.2, rfi_persistent 15.0, radar_chirp 9.6, pulsar 9.5, satellite_pass 8.3, scintillating_star 8.3, frb 7.1, eti_candidate 2.1.
- **Agreement with the earlier v3 safety submission:** 93.3%.

## Files

| File | Purpose |
|---|---|
| `train_pipeline.py` | end-to-end: features → augmentation → (optional CV) → ensemble → validated submission |
| `prep.py` | masking, spike removal, glitch detection, bandpass flattening, robust z |
| `ops.py` | sparse drift-line and de-dispersion operators |
| `features.py` | all cadence features, incl. the global ON/OFF track search |
| `geom.py` | sub-channel ON-track collinearity / drift-consistency geometry |
| `augment.py` | PKS-like augmentation (parameters from unlabeled test statistics) |
| `cv.py`, `stress.py`, `ens_stress.py`, `audit.py`, `make_results.py` | validation, stress test, domain audit, results table |
| `experiment_results.csv` | every measured experiment |
| `confusion_final.png`, `final_*_c.csv` / `final_*_a.csv` | final OOF confusion matrices and per-class metrics |

Data location: pass `--data`, or set `SH_DATA=/path/to/signal-hunters-round2`. The default is `./data`. The helper scripts (`cv.py` etc.) read `SH_DATA` and expect cached features in `./cache`.

## Reproducibility check

`train_pipeline.py` was run from scratch with an empty cache directory on 27 Oct 2026, taking 8.6 min for features and ~21 min for training on 2 CPUs.
- Every regenerated feature table matched the development cache exactly (max abs diff 0.0).
- The output `submission.csv` was **identical** to the uploaded `submission_final.csv` (10,000/10,000 labels).

## Stage 2: CNN blend and self-training (added after the first leaderboard result)

**Leaderboard check.** `submission_final.csv` scored **0.8369** on the public leaderboard. The simulated-PKS stress test predicted 0.842, so it is a reliable proxy for the real shift.

**CNN (`cnn_inputs.py`, `train_cnn.py`)**
- Library: tinygrad, installed from GitHub source (commit `595d397b254eb38b2f0d789a9b38315d740f196a`), because PyPI was unreachable. CPU only.
  - Setup: `git clone https://github.com/tinygrad/tinygrad && git -C tinygrad checkout 595d397b`, then `export TINYGRAD_PATH=$PWD/tinygrad`.
- Input: 12×16×128 per cadence. Six channels of clipped standardized excess power plus six of log-power relative to the OFF-panel reference spectrum.
- Network: 3 conv blocks with BatchNorm, global max+mean pooling, 2 dense layers.
- Training: class-balanced sampling; each cadence is shown clean or PKS-augmented with 50/50 probability; random ±6-channel frequency translation; Adam with cosine schedule, 12 epochs.
- Data: trained on folds 1–4 (seeds 0 and 1); fold 0 held out.
- Alone it scores 0.789 clean / 0.73 PKS-like on fold 0.

**Blend.** 0.65 × HGB ensemble + 0.35 × mean of the 2 CNNs (`make_blend_sub.py 0,1 0.35 submission_final_v2.csv`). On fold 0, HGB alone vs blend:
- clean: 0.8945 → 0.9005;
- PKS-like: 0.8355 → ~0.846 (weights 0.3 and 0.4 give 0.8437 and 0.8487).

**Self-training (`pseudo_sim.py`, `final_pseudo.py`)**
- The unlabeled test cadences whose blend confidence is ≥0.6 receive pseudo-labels.
- The HGB models are retrained with them added, then re-blended with the CNNs.
- No human labels are used.
- Simulation on fold 0 (augmented fold treated as unlabeled test): pseudo-label accuracy 0.91; HGB 0.8320 → 0.8435; blend 0.8432 → 0.8468.

### Final upload: `submission_final_v3.csv`

Build steps, from cached features (see above):
1. `python cnn_inputs.py`
2. `python train_cnn.py 0 12 0` and `python train_cnn.py 1 12 0`
3. `python make_blend_sub.py 0,1 0.35 submission_final_v2.csv` — this also writes `submission_final_v2_proba.npy`, the pseudo-label source.
4. `python final_pseudo.py v4drop 0` and `python final_pseudo.py v3cols 0`
5. Final probabilities: `0.65 × mean(pseudo HGB v4drop, v3cols) + 0.35 × mean(CNN seeds 0, 1)`, then argmax.

Facts:
- Pseudo-labelled test cadences: 8,204 of 10,000 (confidence ≥ 0.6).
- Agreement: 97.3% with v2, 95.8% with `submission_final.csv`.
- Public LB of `submission_final.csv` was 0.8369. The v2/v3 changes were validated only on the simulated-PKS holdout (+0.008 to +0.013, and +0.004 for self-training). No leaderboard score was observed for them.

### Last-hour hybrid: `submission_hybrid_v2.csv`
- A second, independently built submission (`submission_v2.csv`, public LB 0.8571) was combined with this pipeline.
- `agree_pseudo.py`: the HGB models are retrained with pseudo-labels on the 8,778 test cadences where both models agree.
- `final_combine.py`: start from the 0.8571 file's labels, and override with this pipeline's label only where they disagree and the blended confidence is ≥ 0.7 (312 rows).
- Not validated on labels: there was no leaderboard feedback for this hybrid before the deadline.
