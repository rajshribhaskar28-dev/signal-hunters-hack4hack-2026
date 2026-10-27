"""Quantile matching (unsupervised feature alignment) on the PKS-like proxy, fold 0."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import final_model
from final_model import feats, labels, select_features
from train import make_model
yi = labels(); tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[0]
cfg = json.load(open('final_config_v10.json')); cfg['augs'] = [2, 4]
select_features(cfg, rows=tri); g = cfg['groups']
Xc = feats('train', g); X2 = feats('train_aug2', g); X4 = feats('train_aug4', g)

def qmap(src, ref, q=np.linspace(0, 1, 201)):
    out = src.copy()
    for c in src.columns:
        s = src[c].values; r = ref[c].values
        qs = np.quantile(s, q); qr = np.quantile(r, q)
        qs_u, idx = np.unique(qs, return_index=True)          # monotone, handle ties
        if len(qs_u) < 3: continue
        out[c] = np.interp(s, qs_u, qr[idx])
    return out

def run(name, Xtr, ytr, ref):
    m = make_model('hgb', 0, **cfg['hgb']).fit(Xtr, ytr)
    Pv = m.predict_proba(X4.iloc[vai]); Pq = m.predict_proba(qmap(X4.iloc[vai], ref))
    print(f'{name:28s} matched raw {f1_score(yi[vai], Pv.argmax(1), average="macro"):.4f} | matched quantile-mapped {f1_score(yi[vai], Pq.argmax(1), average="macro"):.4f} | clean {f1_score(yi[vai], m.predict_proba(Xc.iloc[vai]).argmax(1), average="macro"):.4f}', flush=True)

run('clean-only model', Xc.iloc[tri], yi[tri], Xc.iloc[tri])
run('clean+aug2+aug4 model', pd.concat([Xc.iloc[tri], X2.iloc[tri], X4.iloc[tri]]), np.concatenate([yi[tri]] * 3), Xc.iloc[tri])
