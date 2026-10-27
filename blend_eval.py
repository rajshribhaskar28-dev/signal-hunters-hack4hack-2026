"""Evaluate HGB (feature model) + CNN probability blends on fold-0 (identical validation rows)."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from train import CLASSES, D, report
tag = sys.argv[1]
tr = pd.read_csv(f'{D}/train_meta.csv'); yi = np.array([CLASSES.index(v) for v in tr.label])
va = np.load('cache/fold0_va.npy'); y = yi[va]
for v in ['cleanval', 'pksval']:
    H = np.load(f'cache/P_v1234sel350_aug2_{v}_f0.npy'); C = np.load(f'cache/P_cnn_{tag}_{v}_f0.npy')
    row = []
    for a in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        P = (1 - a) * H + a * C
        row.append(f'{a}:{f1_score(y, P.argmax(1), average="macro"):.4f}')
    # geometric blend
    G = np.exp(0.5 * np.log(H + 1e-6) + 0.5 * np.log(C + 1e-6))
    print(v, ' '.join(row), '| geo0.5', round(f1_score(y, G.argmax(1), average='macro'), 4))
    print('  per-class F1 cnn ', np.round(f1_score(y, C.argmax(1), average=None), 3))
    print('  per-class F1 hgb ', np.round(f1_score(y, H.argmax(1), average=None), 3))
    if len(sys.argv) > 2:
        a = float(sys.argv[2]); P = (1 - a) * H + a * C
        report(f'blend_hgb_cnn{tag}_a{a}_{v}_f0', y, P, {'model': f'HGB + numpy CNN {tag} (w={a})', 'features': 'groups 1234 (350) + raw z',
               'validation': f'strat fold0/5 {v}', 'seed': 0})
