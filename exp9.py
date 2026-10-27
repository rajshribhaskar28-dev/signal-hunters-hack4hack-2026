import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from final_model import feats, labels, select_features
from train import make_model, report
yi = labels(); F = int(os.environ['FOLD'])
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[F]
cfg = json.load(open('final_config.json')); hgb = dict(max_iter=300, lr=0.08, bins=64)
select_features(cfg, rows=tri)
Xtr = pd.concat([feats('train', '1234').iloc[tri]] + [feats(f'train_aug{a}', '1234').iloc[tri] for a in cfg['augs']])
ytr = np.concatenate([yi[tri]] * (1 + len(cfg['augs'])))
t = time.time(); m = make_model('hgb', 0, **hgb).fit(Xtr, ytr)
for vname, tag in [('cleanval', 'train'), ('matchedval', 'train_aug4')]:
    P = m.predict_proba(feats(tag, '1234').iloc[vai]); np.save(f'cache/P_it300_{vname}_f{F}.npy', P)
    report(f'cap_it300_{vname}_f{F}', yi[vai], P, {'model': f'HGB {hgb}', 'features': 'cal 1234 (350)', 'validation': f'strat fold{F} {vname}', 'seed': 0, 'runtime_s': round(time.time()-t)})
