# Overnight stage: two-pipeline + CNN blend (27–28 Oct)

## Key finding

The two independent pipelines make different mistakes:
- **Pipeline A:** this repo's physics features + HGB, 6 models.
- **Pipeline B:** the team's v9/v11 GBM, repo rajshribhaskar28-dev/signal-hunters-hack4hack-2026.

On pipeline B's fold-0 validation rows (3,600 clean GBT cadences, both out-of-fold), macro-F1:

| Model | Macro-F1 |
|---|---|
| Pipeline B GBM (v9) | 0.912 |
| Pipeline A HGB | 0.903 |
| **50/50 probability average** | **0.931** |

On the 758 rows held out by both splits:

| Blend | Macro-F1 |
|---|---|
| A + B | 0.912 |
| A + B + CNNs at 0.25 | 0.918 |

## CNNs (`npcnn.py`, `train_npcnn.py`)

- **Engine:** pure numpy (im2col + OpenBLAS, ~140 GFLOP/s on 2 CPUs), gradient-checked (max relative error 1e-7).
- **Per-panel CNN** (12 channels × 16 × 128): 0.812 clean / 0.750 PKS-like on fold 0.
- **Global-time CNN** (`GLOBAL=1`):
  - Each panel is placed at its real global-time rows `p·(16+gap)`, giving a 128×128×4 image (excess power, log-power vs OFF reference, ON/OFF indicator, data mask), so drifting tracks are continuous across panels.
  - Result: **0.847 clean / 0.783 PKS-like**, the best single CNN.
- **Pipeline A + all CNNs (w = 0.5):** 0.905 clean / 0.858 PKS-like on fold 0.

## Final probability blend (`build.py`)

`P = 0.375·A + 0.375·B(v11) + 0.25·mean(CNNs)`

- v11 test probabilities were rebuilt with the repo's `make_v11.py`; the result reproduces its `submission.csv` exactly.

## Not validated

- No leaderboard score exists for any blend at the time of writing.
- Weights were chosen on clean GBT validation rows only.
- The two pipelines' PKS-like augmentations use different noise realisations, so they can't be compared jointly.
