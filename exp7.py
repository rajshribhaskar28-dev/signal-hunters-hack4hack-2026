"""Fold-0 comparison of training augmentation strength, scored on clean, harsh (aug2) and
test-matched (aug4) perturbed validation rows. Feature selection inside the fold."""
import sys, json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import final_model
from final_model import feats, labels, select_features
from train import make_model, report
cfg = json.load(open('final_config.json')); cfg['groups'] = '1234'
yi = labels()
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[0]
for spec in sys.argv[1:]:
    name, augs = spec.split(':'); augs = [int(a) for a in augs.split(',') if a]
    cfg['augs'] = augs if augs else [2]
    select_features(cfg, rows=tri)                     # same selection rule as final model
    cfg['augs'] = augs
    Xtr = pd.concat([feats('train', '1234').iloc[tri]] + [feats(f'train_aug{a}', '1234').iloc[tri] for a in augs])
    ytr = np.concatenate([yi[tri]] * (1 + len(augs)))
    t = time.time(); m = make_model('hgb', 0, **cfg['hgb']).fit(Xtr, ytr)
    for vname, tag in [('cleanval', 'train'), ('harshval', 'train_aug2'), ('matchedval', 'train_aug4')]:
        P = m.predict_proba(feats(tag, '1234').iloc[vai]); np.save(f'cache/P_{name}_{vname}_f0.npy', P)
        report(f'{name}_{vname}_f0', yi[vai], P, {'model': 'HGB cal', 'features': 'cal groups 1234 (350)',
               'validation': f'strat fold0 {vname}', 'seed': 0, 'runtime_s': round(time.time() - t)})
