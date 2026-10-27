import json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import final_model
from final_model import feats, labels, select_features
from train import make_model, report
yi = labels(); tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[0]
base = json.load(open('final_config.json'))
for name, nf, hgb in [('cap_it300', 350, dict(max_iter=300, lr=0.08, bins=64)),
                      ('cap_f500', 500, dict(max_iter=150, lr=0.12, bins=64))]:
    cfg = dict(base); cfg['n_features'] = nf; cfg['hgb'] = hgb
    select_features(cfg, rows=tri)
    Xtr = pd.concat([feats('train', '1234').iloc[tri]] + [feats(f'train_aug{a}', '1234').iloc[tri] for a in cfg['augs']])
    ytr = np.concatenate([yi[tri]] * (1 + len(cfg['augs'])))
    t = time.time(); m = make_model('hgb', 0, **hgb).fit(Xtr, ytr)
    for vname, tag in [('cleanval', 'train'), ('matchedval', 'train_aug4')]:
        P = m.predict_proba(feats(tag, '1234').iloc[vai])
        report(f'{name}_{vname}_f0', yi[vai], P, {'model': f'HGB {hgb}', 'features': f'cal 1234 ({nf})', 'validation': f'strat fold0 {vname}', 'seed': 0, 'runtime_s': round(time.time()-t)})
