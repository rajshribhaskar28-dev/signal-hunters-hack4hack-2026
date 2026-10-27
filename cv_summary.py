import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from train import CLASSES, D, cm
tr = pd.read_csv(f'{D}/train_meta.csv'); yi = np.array([CLASSES.index(v) for v in tr.label])
from sklearn.model_selection import StratifiedKFold
folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))
res = {}
for v in ['cleanval', 'pksval']:
    oof = np.zeros((len(yi), 9)); s = []
    for f, (_, va) in enumerate(folds):
        P = np.load(f'cache/P_v1234sel350_aug2_{v}_f{f}.npy'); oof[va] = P
        s.append(f1_score(yi[va], P.argmax(1), average='macro'))
    np.save(f'cache/oof_final_{v}.npy', oof)
    p, r, f1, n = precision_recall_fscore_support(yi, oof.argmax(1), labels=range(9))
    res[v] = pd.DataFrame({'precision': p, 'recall': r, 'f1': f1, 'support': n}, index=CLASSES).round(3)
    print(f'== {v}: fold macro-F1 = {np.round(s,4)}  mean {np.mean(s):.4f} +- {np.std(s):.4f} | pooled OOF {f1_score(yi, oof.argmax(1), average="macro"):.4f}')
    print(res[v].to_string()); print(cm(yi, oof).to_string())
    res[v].to_csv(f'per_class_{v}.csv'); cm(yi, oof).to_csv(f'confusion_{v}.csv')
