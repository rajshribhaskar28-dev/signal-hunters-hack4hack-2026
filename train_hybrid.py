"""Train the hybrid CNN+features network.
python train_hybrid.py <tag> <epochs> <mode: fold0|full> [seed]
fold0: trains on fold-0 training rows; every EVAL epochs scores held-out rows (clean + test-matched
       aug4) and also writes test probabilities from the current weights.
full : trains on all rows; writes test probabilities at the end (and at checkpoints).
Sources sampled per example: clean 0.3, harsh aug2 0.2, matched aug4 0.25, matched aug5 0.25.
"""
import sys, time, json, pickle, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from hybrid import HybridNet
from cnn import Adam, loss_grad, softmax
from train import CLASSES, D
import final_model

tag, EPOCHS, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
EVAL = 4
tr = pd.read_csv(f'{D}/train_meta.csv'); yi = np.array([CLASSES.index(v) for v in tr.label])
SEL = json.load(open('selected_features_final.json')); final_model.SELECT = SEL

def feat(t):
    return final_model.feats(t, '1234').values.astype(np.float32)
Fsrc = [feat('train'), feat('train_aug2'), feat('train_aug4'), feat('train_aug5')]
Ft = feat('test')
med = np.nanmedian(Fsrc[0], 0); iqr = np.nanpercentile(Fsrc[0], 75, 0) - np.nanpercentile(Fsrc[0], 25, 0)
iqr = np.where(iqr > 1e-6, iqr, np.nanstd(Fsrc[0], 0) + 1e-6)
norm = lambda a: np.clip(np.nan_to_num((a - med) / iqr), -6, 6).astype(np.float32)
Fsrc = [norm(a) for a in Fsrc]; Ft = norm(Ft)
Xsrc = [np.load(f'cache/cnn_{t}.npy') for t in ['train', 'train_aug2', 'train_aug4', 'train_aug5']]
Xt = np.load('cache/cnn_test.npy')
if mode == 'fold0':
    tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[0]
else:
    tri, vai = np.arange(len(yi)), None
cnt = np.bincount(yi[tri], minlength=9); cw = (len(tri) / (9 * cnt)).astype(np.float32)
net = HybridNet(nfeat=Ft.shape[1], seed=seed); opt = Adam(net, lr=1e-3, wd=1e-4)
rng = np.random.default_rng(seed); BS = 128; spe = int(np.ceil(len(tri) / BS)); total = EPOCHS * spe; step = 0
PSRC = [0.3, 0.2, 0.25, 0.25]


def predict(X, Fm, idx=None, bs=500):
    idx = np.arange(len(X)) if idx is None else idx
    out = np.zeros((len(idx), 9), np.float32)
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]; out[s:s + bs] = softmax(net.forward(X[j].astype(np.float32), Fm[j]))
    return out


best = -1
for ep in range(EPOCHS):
    t = time.time(); L = 0
    for it in range(spe):
        ids = rng.choice(tri, BS); src = rng.choice(4, BS, p=PSRC)
        x = np.empty((BS, 16, 128, 6), np.float32); f = np.empty((BS, Ft.shape[1]), np.float32)
        for k in range(4):
            m = src == k
            if m.any(): x[m] = Xsrc[k][ids[m]]; f[m] = Fsrc[k][ids[m]]
        sh = rng.integers(-8, 9)
        if sh:
            x = np.roll(x, sh, axis=2)
            if sh > 0: x[:, :, :sh] = 0
            else: x[:, :, sh:] = 0
        lr = 2e-5 + 0.5 * (1e-3 - 2e-5) * (1 + np.cos(np.pi * step / total)); step += 1
        loss, g = loss_grad(net.forward(x, f), yi[ids], cw)
        net.backward(g); opt.step(lr); L += loss
    msg = f'ep {ep} loss {L / spe:.4f} {time.time() - t:.0f}s lr {lr:.2e}'
    if (ep + 1) % EVAL == 0 or ep == EPOCHS - 1:
        if mode == 'fold0':
            Pc = predict(Xsrc[0], Fsrc[0], vai); Pm = predict(Xsrc[2], Fsrc[2], vai)
            fc = f1_score(yi[vai], Pc.argmax(1), average='macro'); fm = f1_score(yi[vai], Pm.argmax(1), average='macro')
            msg += f' | clean {fc:.4f} matched {fm:.4f}'
            np.save(f'cache/P_hyb_{tag}_cleanval_f0_ep{ep}.npy', Pc); np.save(f'cache/P_hyb_{tag}_matchedval_f0_ep{ep}.npy', Pm)
            if fm > best:
                best = fm
                np.save(f'cache/P_hyb_{tag}_cleanval_f0.npy', Pc); np.save(f'cache/P_hyb_{tag}_matchedval_f0.npy', Pm)
                np.save(f'cache/testP_hyb_{tag}.npy', predict(Xt, Ft)); pickle.dump(net, open(f'cache/hyb_{tag}.pkl', 'wb'))
                msg += ' *saved*'
        else:
            np.save(f'cache/testP_hyb_{tag}.npy', predict(Xt, Ft)); pickle.dump(net, open(f'cache/hyb_{tag}.pkl', 'wb'))
    print(msg, flush=True)
print('done', flush=True)
