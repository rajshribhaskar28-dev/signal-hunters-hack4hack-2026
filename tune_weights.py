"""Per-class probability weights for macro-F1 (coordinate ascent on a log grid).
Fit on one validation view, check on the other (and vice versa) to detect overfitting."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from train import CLASSES, D

def mf1(y, P, w): return f1_score(y, (P * w).argmax(1), average='macro')

def fit_w(y, P, rounds=3, grid=np.exp(np.linspace(-0.7, 0.7, 15))):
    w = np.ones(P.shape[1])
    for _ in range(rounds):
        for c in range(P.shape[1]):
            best = (mf1(y, P, w), w[c])
            for g in grid:
                w2 = w.copy(); w2[c] = g
                s = mf1(y, P, w2)
                if s > best[0] + 1e-4: best = (s, g)
            w[c] = best[1]
    return w

if __name__ == '__main__':
    name = sys.argv[1]
    tr = pd.read_csv(f'{D}/train_meta.csv'); yi = np.array([CLASSES.index(v) for v in tr.label])
    va = np.load('cache/va_f0.npy'); y = yi[va]
    Pc = np.load(f'cache/P_{name}_cleanval_f0.npy'); Pp = np.load(f'cache/P_{name}_pksval_f0.npy')
    one = np.ones(9)
    print('base  clean %.4f pks %.4f' % (mf1(y, Pc, one), mf1(y, Pp, one)))
    wp = fit_w(y, Pp); wc = fit_w(y, Pc)
    print('w fit on pks  -> clean %.4f pks %.4f' % (mf1(y, Pc, wp), mf1(y, Pp, wp)), np.round(wp, 2))
    print('w fit on clean-> clean %.4f pks %.4f' % (mf1(y, Pc, wc), mf1(y, Pp, wc)), np.round(wc, 2))
    # 2-fold check within pks view: fit on half, score other half
    rng = np.random.default_rng(0); idx = rng.permutation(len(y)); h = len(y) // 2
    a, b = idx[:h], idx[h:]
    for f, s in [(a, b), (b, a)]:
        w = fit_w(y[f], Pp[f])
        print('pks half-split: base %.4f -> weighted %.4f' % (mf1(y[s], Pp[s], one), mf1(y[s], Pp[s], w)))
