import os, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from final_model import feats, labels, select_features
from train import make_model, report
yi = labels(); cfg = json.load(open('final_config_v10.json')); g = cfg['groups']
for F in [int(x) for x in os.environ['FOLDS'].split(',')]:
    tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[F]
    select_features(cfg, rows=tri)
    Xtr = pd.concat([feats('train', g).iloc[tri]] + [feats(f'train_aug{a}', g).iloc[tri] for a in cfg['augs']])
    t = time.time(); m = make_model('hgb', 0, **cfg['hgb']).fit(Xtr, np.concatenate([yi[tri]] * (1 + len(cfg['augs']))))
    for vname, tag in [('cleanval', 'train'), ('matchedval', 'train_aug4')]:
        P = m.predict_proba(feats(tag, g).iloc[vai]); np.save(f'cache/P_v10cv_{vname}_f{F}.npy', P)
        report(f'v10cv_{vname}_f{F}', yi[vai], P, {'model': f"HGB {cfg['hgb']}", 'features': f'groups {g} (350)', 'validation': f'strat fold{F} {vname}', 'seed': 0, 'runtime_s': round(time.time()-t)})
