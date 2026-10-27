"""Transductive pseudo-label validation on the PKS-like proxy.
Target = held-out fold rows under PKS-like perturbation, treated as UNLABELED.
Their true labels are used only for scoring."""
import json, sys, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import final_model
from final_model import feats, labels
from train import make_model, report, CLASSES
final_model.SELECT = json.load(open('selected_features_350.json'))
yi = labels(); Xc = feats('train', '1234'); Xa = feats('train_aug2', '1234')
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yi))[0]
Xtr = pd.concat([Xc.iloc[tri], Xa.iloc[tri]]); ytr = np.concatenate([yi[tri], yi[tri]])
Xt = Xa.iloc[vai]; yt = yi[vai]
kw = dict(max_iter=150, lr=0.12, bins=64)
P = np.load('cache/P_v1234sel350_aug2_pksval_f0.npy')   # round-0 model = same config & rows
for thr in [float(x) for x in sys.argv[1].split(',')]:
    Pr = P.copy()
    for rnd in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1):
        conf = Pr.max(1); sel = conf >= thr
        pl = Pr.argmax(1)[sel]
        acc = (pl == yt[sel]).mean()
        t = time.time()
        m = make_model('hgb', 0, **kw).fit(pd.concat([Xtr, Xt[sel]]), np.concatenate([ytr, pl]))
        Pr = m.predict_proba(Xt)
        report(f'pl_thr{thr}_r{rnd+1}_pksval_f0', yt, Pr, {'model': 'HGB + pseudo-labels', 'features': 'groups 1234 (350)',
               'validation': f'transductive fold0 pksval; {sel.sum()} PL rows acc {acc:.3f}', 'seed': 0,
               'runtime_s': round(time.time() - t)})
