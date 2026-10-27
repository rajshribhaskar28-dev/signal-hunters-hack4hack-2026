"""Cross-validated training / evaluation harness.

python train.py <experiment_name> [--aug] [--drop-prefix ...]
"""
import sys, time, json, os, argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

D = os.environ.get('SH_DATA', 'data')
CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
           'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']   # priority order
# features that encode telescope/acquisition rather than physics
EXCLUDE = ['slew_gap_rows']


def load(tag):
    return pd.read_pickle(f'cache/feat_{tag}.pkl')


def make_model(kind, seed, **kw):
    if kind == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=kw.get('max_iter', 400), learning_rate=kw.get('lr', 0.06),
            max_leaf_nodes=kw.get('leaves', 31), min_samples_leaf=kw.get('msl', 20),
            l2_regularization=kw.get('l2', 1.0), class_weight=kw.get('cw', 'balanced'),
            max_bins=kw.get('bins', 255), early_stopping=False, random_state=seed)
    if kind == 'et':
        return ExtraTreesClassifier(n_estimators=kw.get('n', 500), min_samples_leaf=kw.get('msl', 2),
                                    max_features=kw.get('mf', 'sqrt'), class_weight='balanced',
                                    n_jobs=2, random_state=seed)
    raise ValueError(kind)


def cv_oof(F, y, kind='hgb', seed=0, n_splits=5, F_val=None, F_extra=None, **kw):
    """OOF probabilities. F_val: alternative representation of the same rows used for scoring
    (e.g. PKS-like augmented copy). F_extra: extra training rows aligned with F (augmented copy),
    whose fold rows are excluded from training for the matching validation fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(F), len(CLASSES)))
    oof_val = None if F_val is None else np.zeros_like(oof)
    yi = np.array([CLASSES.index(v) for v in y])
    for tr, va in skf.split(F, yi):
        Xtr, ytr = F.iloc[tr], yi[tr]
        if F_extra is not None:
            Xtr = pd.concat([Xtr, F_extra.iloc[tr]]); ytr = np.concatenate([ytr, yi[tr]])
        m = make_model(kind, seed, **kw).fit(Xtr, ytr)
        oof[va] = m.predict_proba(F.iloc[va])
        if F_val is not None:
            oof_val[va] = m.predict_proba(F_val.iloc[va])
    return oof, oof_val, yi


def report(name, yi, P, extra=None, log='experiment_results.csv'):
    pred = P.argmax(1)
    mf = f1_score(yi, pred, average='macro')
    pc = f1_score(yi, pred, average=None, labels=range(len(CLASSES)))
    row = {'experiment': name, 'macro_f1': round(mf, 4)}
    row.update({f'f1_{c}': round(v, 4) for c, v in zip(CLASSES, pc)})
    if extra: row.update(extra)
    df = pd.DataFrame([row])
    df.to_csv(log, mode='a', header=not os.path.exists(log), index=False)
    print(f'{name}: macroF1={mf:.4f} | ' + ' '.join(f'{c[:6]}={v:.3f}' for c, v in zip(CLASSES, pc)))
    return mf


def cm(yi, P):
    c = confusion_matrix(yi, P.argmax(1), labels=range(len(CLASSES)))
    return pd.DataFrame(c, index=[x[:10] for x in CLASSES], columns=[x[:6] for x in CLASSES])
