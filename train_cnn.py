"""Train the numpy CNN.
python train_cnn.py fold0 <epochs> <tag>   -> trains on fold-0 train rows (clean + aug2 + aug3 copies),
                                             reports clean / PKS-like val each epoch, saves val probs
python train_cnn.py full <epochs> <tag>    -> trains on all rows, saves test probs
"""
import sys, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from cnn import Net, Adam, loss_grad, softmax
from train import CLASSES, D

mode, EPOCHS, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
WID = tuple(int(w) for w in sys.argv[4].split(',')) if len(sys.argv) > 4 else (24, 48, 64, 96)
tr = pd.read_csv(f'{D}/train_meta.csv'); yi = np.array([CLASSES.index(v) for v in tr.label])
srcs = [np.load('cache/cnn_train.npy'), np.load('cache/cnn_train_aug2.npy'), np.load('cache/cnn_train_aug3.npy')]
if mode == 'fold0':
    tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[0]
else:
    tri, vai = np.arange(len(yi)), None
    Xte = np.load('cache/cnn_test.npy')
cnt = np.bincount(yi[tri], minlength=9); cw = (len(tri) / (9 * cnt)).astype(np.float32)
net = Net(widths=WID, seed=0); opt = Adam(net, lr=2e-3)
rng = np.random.default_rng(0)
BS = 128
steps_per_epoch = int(np.ceil(len(tri) * 1.5 / BS))          # 1.5 passes worth of mixed-source samples
total = EPOCHS * steps_per_epoch; step = 0


def predict(X, idx=None, bs=500):
    idx = np.arange(len(X)) if idx is None else idx
    out = np.zeros((len(idx), 9), np.float32)
    for s in range(0, len(idx), bs):
        out[s:s + bs] = softmax(net.forward(X[idx[s:s + bs]].astype(np.float32)))
    return out


def batch(ids, src):
    x = np.empty((len(ids), 16, 128, 6), np.float32)
    for k in range(3):
        m = src == k
        if m.any(): x[m] = srcs[k][ids[m]]
    # augmentation: random frequency reversal (flips drift sign; dispersion handled by keeping p=0 for
    # broadband classes would need labels - instead keep physical orientation) -> only small freq shifts
    sh = rng.integers(-8, 9)
    if sh:
        x = np.roll(x, sh, axis=2)
        if sh > 0: x[:, :, :sh] = 0
        else: x[:, :, sh:] = 0
    return x


for ep in range(EPOCHS):
    t = time.time(); L = 0
    for it in range(steps_per_epoch):
        ids = rng.choice(tri, BS)
        src = rng.choice(3, BS, p=[0.4, 0.3, 0.3])
        x = batch(ids, src)
        lr = 1e-4 + 0.5 * (2e-3 - 1e-4) * (1 + np.cos(np.pi * step / total)); step += 1
        logits = net.forward(x, train=True)
        loss, g = loss_grad(logits, yi[ids], cw)
        net.backward(g); opt.step(lr); L += loss
    msg = f'ep {ep} loss {L / steps_per_epoch:.4f} {time.time() - t:.0f}s'
    if mode == 'fold0':
        Pc = predict(srcs[0], vai); Pp = predict(srcs[1], vai)
        msg += f' | clean {f1_score(yi[vai], Pc.argmax(1), average="macro"):.4f} pks {f1_score(yi[vai], Pp.argmax(1), average="macro"):.4f}'
        np.save(f'cache/P_cnn_{tag}_cleanval_f0.npy', Pc); np.save(f'cache/P_cnn_{tag}_pksval_f0.npy', Pp)
    print(msg, flush=True)
if mode == 'full':
    np.save(f'cache/testP_cnn_{tag}.npy', predict(Xte))
import pickle; pickle.dump(net, open(f'cache/cnn_{tag}.pkl', 'wb'))
print('done', flush=True)
