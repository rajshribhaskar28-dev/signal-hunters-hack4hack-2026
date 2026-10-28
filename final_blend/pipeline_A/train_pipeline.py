"""Signal Hunters (Hack4Hack 2026 Round 2) — end-to-end reproducible pipeline.

Usage:
    python train_pipeline.py --data /path/to/signal-hunters-round2 [--cv] [--out submission.csv]

Steps
 1. Physics-informed feature extraction (features.py) for train and test (cached in ./cache).
 2. PKS-like augmentation of the training cadences (augment.py); parameters estimated from
    UNLABELED test statistics only (bandpass median profile, noise-width / tail quantiles, gain spread,
    dropped/glitch row rates). No test labels exist or are used.
 3. Optional 5-fold stratified CV (seed 42) reporting macro-F1 on clean GBT holdout and on the
    PKS-augmented version of the same holdout rows (domain-shift stress test).
 4. Final HistGradientBoosting ensemble (3 seeds, probability average) trained on clean + 2 augmented copies.
 5. Writes submission.csv in sample_submission order and validates its schema.
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from features import cadence_features
from augment import pksify, bandpass_profiles

CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
           'rfi_intermittent', 'rfi_persistent', 'noise']
AUG_SEEDS = [2, 3]
# absolute noise-floor statistics found telescope-dependent in the train-vs-test audit; their ON-OFF '_diff'
# contrasts are kept. Dropping them did not change CV (E6) and reduces reliance on receiver-specific quirks.
DROP = ['f3_', 'f5_', 'f8_', 'p99_', 'max_', 'top32_', 'std_', 'specnoise_', 'row_var_', 'devmax_']
NEW_V4 = ['trk_']   # global-track occupancy features added in v4
def feature_sets(columns):
    v4drop = [c for c in columns if c.endswith('_diff') or not any(c.startswith(d) for d in DROP)]
    v3cols = [c for c in columns if not any(c.startswith(d) for d in NEW_V4)]
    return {'v4drop': v4drop, 'v3cols': v3cols}
USE_SETS = ['v4drop', 'v3cols']
MODEL_SEEDS = [0, 1, 2]
BATCH = 200


def extract(X, gaps, n_jobs=2, aug_seed=None, delta=None):
    def run(s):
        v = np.asarray(X[s:s + BATCH])
        if aug_seed is not None:
            v = pksify(v, np.random.default_rng(aug_seed * 100000 + s), delta)
        return cadence_features(v, gaps[s:s + BATCH])
    F = pd.concat(Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in range(0, len(X), BATCH)), ignore_index=True)
    F['slew_gap_rows'] = gaps
    return F


def cached(path, fn):
    if os.path.exists(path):
        return pd.read_pickle(path)
    F = fn(); F.to_pickle(path); return F


def make_model(seed):
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, class_weight='balanced', early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=30, random_state=seed)


def tune_bias(P, y, iters=3, grid=np.linspace(-1.5, 1.5, 31)):
    """coordinate ascent on per-class additive log-prob offsets to maximise macro-F1"""
    L = np.log(P + 1e-9); b = np.zeros(P.shape[1])
    best = f1_score(y, (L + b).argmax(1), average='macro')
    for _ in range(iters):
        for c in range(P.shape[1]):
            for v in grid:
                bb = b.copy(); bb[c] = v
                s = f1_score(y, (L + bb).argmax(1), average='macro')
                if s > best + 1e-5: best, b = s, bb
    return b, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('SH_DATA', 'data'))
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--cv', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    D = a.data
    Xtr = np.load(f'{D}/train_cadences.npy', mmap_mode='r'); Xte = np.load(f'{D}/test_cadences.npy', mmap_mode='r')
    tr = pd.read_csv(f'{D}/train_meta.csv'); te = pd.read_csv(f'{D}/test_meta.csv')
    ss = pd.read_csv(f'{D}/sample_submission.csv')
    assert len(tr) == len(Xtr) and len(te) == len(Xte) and (ss.id.values == te.id.values).all()
    y = tr.label.map({c: i for i, c in enumerate(CLASSES)}).values
    t0 = time.time()
    delta_path = f'{a.cache}/bp_delta.npy'
    if not os.path.exists(delta_path):
        np.save(delta_path, bandpass_profiles(Xte) - bandpass_profiles(Xtr))
    delta = np.load(delta_path)
    Ftr = cached(f'{a.cache}/F_train_v5.pkl', lambda: extract(Xtr, tr.slew_gap_rows.values))
    Fte = cached(f'{a.cache}/F_test_v5.pkl', lambda: extract(Xte, te.slew_gap_rows.values))
    Faug = [cached(f'{a.cache}/F_trainaug{s}_v5.pkl', lambda s=s: extract(Xtr, tr.slew_gap_rows.values, aug_seed=s, delta=delta))
            for s in AUG_SEEDS]
    print(f'features ready {time.time() - t0:.0f}s', Ftr.shape, flush=True)
    sets = {k: v for k, v in feature_sets(list(Ftr.columns)).items() if k in USE_SETS}
    cols = sets[USE_SETS[0]]
    if a.cv:
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        Pc = np.zeros((len(y), 9)); Pa = np.zeros((len(y), 9))
        for k, (i, j) in enumerate(skf.split(Ftr, y)):
            for name, cs in sets.items():
                m = make_model(0)
                m.fit(pd.concat([Ftr.iloc[i][cs]] + [F.iloc[i][cs] for F in Faug]), np.concatenate([y[i]] * (1 + len(Faug))))
                Pc[j] += m.predict_proba(Ftr.iloc[j][cs]) / len(sets); Pa[j] += m.predict_proba(Faug[0].iloc[j][cs]) / len(sets)
            print('fold', k, flush=True)
        np.save(f'{a.cache}/oof_final_clean.npy', Pc); np.save(f'{a.cache}/oof_final_aug.npy', Pa)
        print('OOF macro-F1 clean %.4f | PKS-aug %.4f' % (f1_score(y, Pc.argmax(1), average='macro'),
                                                          f1_score(y, Pa.argmax(1), average='macro')))
    # per-class offsets were evaluated (tune_bias on half of OOF, scored on the other half) and gave no
    # reproducible gain (+-0.001), so the final model uses plain argmax of the averaged probabilities.
    b = np.zeros(len(CLASSES))
    # final models on all data
    Xall = pd.concat([Ftr] + Faug); yall = np.concatenate([y] * (1 + len(Faug)))
    P = np.zeros((len(Fte), 9)); nm = 0
    for name, cs in sets.items():
        for sd in MODEL_SEEDS:
            m = make_model(sd); m.fit(Xall[cs], yall); P += m.predict_proba(Fte[cs]); nm += 1
            print('model', name, 'seed', sd, 'done', flush=True)
    P /= nm
    np.save(f'{a.cache}/test_proba.npy', P)
    pred = (np.log(P + 1e-9) + b).argmax(1)
    sub = pd.DataFrame({'id': te.id.values, 'label': np.array(CLASSES)[pred]})
    sub = ss[['id']].merge(sub, on='id', how='left')
    # validation of submission schema
    assert list(sub.columns) == ['id', 'label'] and len(sub) == 10000
    assert sub.id.is_unique and (sub.id.values == ss.id.values).all()
    assert sub.label.isin(CLASSES).all() and sub.label.notna().all()
    sub.to_csv(a.out, index=False)
    print('wrote', a.out, sub.label.value_counts().to_dict(), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
