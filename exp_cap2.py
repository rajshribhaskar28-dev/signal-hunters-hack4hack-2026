import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from final_model import feats, labels, select_features
from train import make_model, report
yi = labels(); F = int(os.environ.get('FOLD', '0'))
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[F]
cfg = json.load(open('final_config_v10.json')); cfg['augs'] = [2, 4]; g = cfg['groups']
hgb = dict(max_iter=500, lr=0.06, bins=64, leaves=47, msl=30)
select_features(cfg, rows=tri)
Xtr = pd.concat([feats('train', g).iloc[tri]] + [feats(f'train_aug{a}', g).iloc[tri] for a in cfg['augs']])
t = time.time(); m = make_model('hgb', 0, **hgb).fit(Xtr, np.concatenate([yi[tri]] * 3))
for vname, tag in [('cleanval', 'train'), ('matchedval', 'train_aug4')]:
    P = m.predict_proba(feats(tag, g).iloc[vai]); np.save(f'cache/P_cap2_{vname}_f{F}.npy', P)
    report(f'cap2_g12346_a24_{vname}_f{F}', yi[vai], P, {'model': f'HGB {hgb}', 'features': 'groups 12346 (350)', 'validation': f'strat fold{F} {vname}', 'seed': 0, 'runtime_s': round(time.time()-t)})
