"""Signal Hunters (Hack4Hack 2026, Round 2) - end-to-end reproducible pipeline.

Regenerates the submitted CSV from the original competition files:
    python train_pipeline.py --data /path/to/signal-hunters-round2 [--skip-features] [--cv]

Steps
 1. Feature extraction (cached in ./cache): four physics-informed groups
      features.py  : de-Doppler drift search (slow / fast / steep), global ON/OFF track test in
                     global time p*(16+slew_gap_rows)+r, f^-2 dedispersion search, pixel/texture stats
      features2.py : cross-panel (leave-one-panel-out) bandpass -> broad ON-only emission
      features3.py : explicit ETI track-consistency refinement (local offsets, line-fit residual)
      features4.py : line-robust texture + matched 2-D power-spectrum scintillation detector
    for: train (GBT), test (PKS), and PKS-like augmented copies of train (augment.py; seeds 2,3)
 2. Optional fold-0 validation on clean GBT and on PKS-like perturbed GBT (--cv)
 3. HistGradientBoosting (balanced class weights) fit on clean + augmented train
 4. Predict test, apply class-probability weights tuned on validation, write submission.csv
Only the provided training data are used; test data are used unlabeled only (to estimate
augmentation noise/bandpass statistics). Seeds are fixed.
"""
import argparse, os, subprocess, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CFG = json.load(open(os.path.join(HERE, 'final_config.json')))


def run_features(data):
    env = dict(os.environ, SH_DATA=data)
    jobs = []
    for split, aug in [('train', None), ('test', None)] + [('train', a) for a in CFG['augs']]:
        for mod in ['features', 'features2', 'features3', 'features4']:
            tag = split if aug is None else f'{split}_aug{aug}'
            pre = {'features': 'feat', 'features2': 'feat2', 'features3': 'feat3', 'features4': 'feat4'}[mod]
            if os.path.exists(os.path.join(HERE, 'cache', f'{pre}_{tag}.pkl')):
                continue
            cmd = [sys.executable, os.path.join(HERE, f'{mod}.py'), split] + ([str(aug)] if aug is not None else [])
            print('running', ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=HERE, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('SH_DATA', 'data'))
    ap.add_argument('--skip-features', action='store_true')
    ap.add_argument('--cv', action='store_true')
    a = ap.parse_args()
    os.environ['SH_DATA'] = a.data
    os.chdir(HERE)
    if not a.skip_features:
        run_features(a.data)
    from final_model import fit_predict, validate
    if a.cv:
        validate(CFG)
    fit_predict(CFG, out='submission.csv')


if __name__ == '__main__':
    main()
