import time, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from train import *
tr = pd.read_csv(f'{D}/train_meta.csv')
yi = np.array([CLASSES.index(v) for v in tr.label])
F1, F1a = load('train').drop(columns=EXCLUDE), load('train_aug2').drop(columns=EXCLUDE)
F2, F2a = pd.read_pickle('cache/feat2_train.pkl'), pd.read_pickle('cache/feat2_train_aug2.pkl')
Fc, Fa = pd.concat([F1, F2], axis=1), pd.concat([F1a, F2a], axis=1)
F3, F3a = pd.read_pickle('cache/feat3_train.pkl'), pd.read_pickle('cache/feat3_train_aug2.pkl')
Gc, Ga = pd.concat([Fc, F3], axis=1), pd.concat([Fa, F3a], axis=1)
fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(F1, yi))[fold]
kw = dict(max_iter=150, lr=0.12, bins=64)
configs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['v1_clean', 'v12_clean', 'v12_cleanaug']
for name in configs:
    feats = (F1, F1a) if name.startswith('v1_') else (Gc, Ga) if name.startswith('v123') else (Fc, Fa)
    Xc, Xa = feats
    if name.endswith('cleanaug'):
        Xtr = pd.concat([Xc.iloc[tri], Xa.iloc[tri]]); ytr = np.concatenate([yi[tri], yi[tri]])
    else:
        Xtr, ytr = Xc.iloc[tri], yi[tri]
    t = time.time()
    m = make_model('hgb', 0, **kw).fit(Xtr, ytr)
    rt = round(time.time() - t)
    for vname, Xv in [('cleanval', Xc), ('pksval', Xa)]:
        P = m.predict_proba(Xv.iloc[vai]); np.save(f'cache/P_{name}_{vname}_f{fold}.npy', P)
        report(f'{name}_{vname}_f{fold}', yi[vai], P, {'model': 'HGB it150 lr.12 bins64', 'features': f'{name} ({Xc.shape[1]})',
               'validation': f'strat fold{fold}/5 {vname}', 'seed': 0, 'runtime_s': rt})
    imp_name = name
np.save(f'cache/va_f{fold}.npy', vai)
