# Final submission: 17_final_v2x4_G50.csv (best public LB 0.8906 for the same recipe with 2 v2 seeds)

Log-space blend of:
- 0.20 Pipeline A: physics features + HistGradientBoosting (`pipeline_A/train_pipeline.py`).
- 0.20 Pipeline B: team v11 (repo root, `make_v11.py`).
- 0.50 GPU global-time CNN:
  - `cnn_and_blend/gpu_global_cnn.py`: 4 seeds (v1).
  - `cnn_and_blend/gpu_global_cnn_v2.py`: 4 seeds, with 3 PKS-like augmentations and confident test pseudo-labels.
  - Mixed 1/3 v1 + 2/3 v2. Run on a Colab T4.
- 0.10 CPU CNNs (`cnn_and_blend/train_npcnn.py`, `npcnn.py`).

The final file is built by `cnn_and_blend/final17.py`.

Uses only the provided competition data. Pseudo-labels come from model predictions, with no human labelling.

## Submission files (`submissions/`)
- `15_logblend_v2heavy_G50.csv`: public LB **0.8906** (best confirmed score).
- `17_final_v2x4_G50.csv`: the same recipe with 4 v2 GPU seeds instead of 2 (final upload; score pending at commit time).
