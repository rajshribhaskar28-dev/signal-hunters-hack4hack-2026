"""Feature group 5: per-line panel-occupancy inventory.

Priority: a cadence with an intermittent line AND a persistent line is rfi_intermittent.
Group 1 only characterises the strongest line, so a weaker intermittent line next to a bright
persistent one is missed. Here every line peak (slow-drift search, any panel) is matched
against all six panels at the same panel-mid channel (+-2 ch, any slow drift), and the
cadence is described by how many lines occupy 1..6 panels, ON-only / OFF-including patterns.
python features5.py <train|test> [aug_seed]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from scipy.ndimage import maximum_filter1d
from prep import preprocess
from drift import drift_sums
from features import SLOW, PAD, D, ON, OFF

warnings.filterwarnings('ignore')
ZERO = np.abs(SLOW) <= 0.25


def extract5(u8):
    B = u8.shape[0]
    z, valid, _, _ = preprocess(u8)
    S, _ = drift_sums(z, valid, SLOW[ZERO | (np.abs(SLOW) <= 1.5)], PAD)
    prof = S.max(2)                                                   # B,6,W best over drift per channel
    W = prof.shape[-1]
    near = maximum_filter1d(prof, 5, axis=-1)                         # presence within +-2 channels
    out = {}
    for thr, nm in [(6.0, 'h'), (4.5, 'l')]:
        pk = (prof >= maximum_filter1d(prof, 7, axis=-1)) & (prof > thr)   # B,6,W peaks
        lit = near > (thr - 1.5)                                       # B,6,W panel has a line here
        npan = lit.sum(1)                                              # B,W panels lit at channel
        on_n = lit[:, ON].sum(1); off_n = lit[:, OFF].sum(1)
        any_pk = pk.any(1)                                             # B,W line peak in some panel
        for k in range(1, 7):
            out[f'inv{nm}_n{k}'] = (any_pk & (npan == k)).sum(1)
        out[f'inv{nm}_n5plus'] = (any_pk & (npan >= 5)).sum(1)
        out[f'inv{nm}_n1to4'] = (any_pk & (npan >= 1) & (npan <= 4)).sum(1)
        out[f'inv{nm}_onlyON'] = (any_pk & (off_n == 0) & (on_n >= 1)).sum(1)
        out[f'inv{nm}_onANDoff_partial'] = (any_pk & (off_n >= 1) & (on_n >= 1) & (npan <= 4)).sum(1)
        out[f'inv{nm}_onlyOFF'] = (any_pk & (on_n == 0) & (off_n >= 1)).sum(1)
        # strength of the strongest partial (1..4 panels) line vs strongest 5+ line
        pmax = prof.max(1)
        out[f'inv{nm}_partial_max'] = np.where(any_pk & (npan <= 4), pmax, 0).max(1)
        out[f'inv{nm}_full_max'] = np.where(any_pk & (npan >= 5), pmax, 0).max(1)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract5(u8))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat5_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
