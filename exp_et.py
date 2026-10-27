import json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import final_model
from final_model import feats, labels
from train import make_model, report
final_model.SELECT = json.load(open('selected_features_350.json'))
yi = labels(); Xc = feats('train', '1234'); Xa = feats('train_aug2', '1234')
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yi))[0]
X = pd.concat([Xc.iloc[tri], Xa.iloc[tri]]); y = np.concatenate([yi[tri], yi[tri]])
t = time.time()
m = make_model('et', 0, n=400, msl=2, mf=0.15).fit(X, y)
for v, Xv in [('cleanval', Xc), ('pksval', Xa)]:
    P = m.predict_proba(Xv.iloc[vai]); np.save(f'cache/P_et_sel350_{v}_f0.npy', P)
    report(f'et_sel350_{v}_f0', yi[vai], P, {'model': 'ExtraTrees 400 msl2 mf.15', 'features': 'groups 1234 (350)',
           'validation': f'strat fold0/5 {v}', 'seed': 0, 'runtime_s': round(time.time() - t)})
    Ph = np.load(f'cache/P_v1234sel350_aug2_{v}_f0.npy')
    for a in [0.2, 0.35, 0.5]:
        print(v, 'blend hgb/et', a, round(f1_score(yi[vai], ((1 - a) * Ph + a * P).argmax(1), average='macro'), 4))
