import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from final_model import feats, labels, select_features
from train import make_model, report
yi = labels(); F = int(os.environ.get('FOLD', '0'))
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[F]
cfg = json.load(open('final_config.json')); cfg['groups'] = os.environ.get('GROUPS', '1234r'); cfg['augs'] = [2, 4]
select_features(cfg, rows=tri)
g = cfg['groups']
Xtr = pd.concat([feats('train', g).iloc[tri]] + [feats(f'train_aug{a}', g).iloc[tri] for a in cfg['augs']])
ytr = np.concatenate([yi[tri]] * (1 + len(cfg['augs'])))
t = time.time(); m = make_model('hgb', 0, **cfg['hgb']).fit(Xtr, ytr)
for vname, tag in [('cleanval', 'train'), ('matchedval', 'train_aug4')]:
    P = m.predict_proba(feats(tag, g).iloc[vai]); np.save(f"cache/P_g{cfg['groups']}_a24_{vname}_f{F}.npy", P)
    report(f"g{cfg['groups']}_a24_it300_{vname}_f{F}", yi[vai], P, {'model': f"HGB {cfg['hgb']}", 'features': f"groups {cfg['groups']} (350)", 'validation': f'strat fold{F} {vname}', 'seed': 0, 'runtime_s': round(time.time()-t)})
