"""Recompute features for slices of the raw data and compare to the cache (reproducibility check)."""
import numpy as np, pandas as pd
from features import extract, D
from features2 import extract2
from features3 import extract3
from features4 import extract4
from augment import pks_like
ok = True
for split, s, aug in [('train', 4000, None), ('test', 8000, None), ('train', 1200, 2)]:
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r'); gaps = pd.read_csv(f'{D}/{split}_meta.csv').slew_gap_rows.values
    u8 = np.asarray(X[s:s + 200])
    if aug is not None: u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
    tag = split if aug is None else f'{split}_aug{aug}'
    for pre, fn in [('feat', lambda: extract(u8, gaps[s:s+200])), ('feat2', lambda: extract2(u8)),
                    ('feat3', lambda: extract3(u8, gaps[s:s+200])), ('feat4', lambda: extract4(u8))]:
        new = fn(); old = pd.read_pickle(f'cache/{pre}_{tag}.pkl').iloc[s:s + 200].reset_index(drop=True)
        d = np.nanmax(np.abs(new.values - old[new.columns].values))
        ok &= d < 1e-4
        print(tag, pre, 'max abs diff', d)
print('REPRODUCIBLE' if ok else 'MISMATCH')
