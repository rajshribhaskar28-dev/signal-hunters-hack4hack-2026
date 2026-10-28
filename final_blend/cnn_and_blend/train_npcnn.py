"""Train the numpy CNN (npcnn.py). Resumable; saves holdout/test predictions periodically.
usage: python train_npcnn.py SEED EPOCHS HOLD [width]"""
import sys, os, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npcnn import Net, Net2, Adam, softmax_ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
SEED, EPOCHS, HOLD = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
WIDTH = tuple(int(w) for w in sys.argv[4].split(',')) if len(sys.argv) > 4 else (32, 64, 96, 128)
GLOBAL = os.environ.get('GLOBAL') == '1'
TAG = ('g' if GLOBAL else '') + f'np_s{SEED}_h{HOLD}' + ('_pl' if os.environ.get('PSEUDO') else '')
C = '/home/claude/work/cache'; D = os.environ.get('SH_DATA', '/home/claude/data/signal-hunters-round2')
CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
           'rfi_intermittent', 'rfi_persistent', 'noise']
y = pd.read_csv(f'{D}/train_meta.csv').label.map({c: i for i, c in enumerate(CLASSES)}).values
Xc = np.load(f'{C}/cnn_train.npy', mmap_mode='r'); Xa = np.load(f'{C}/cnn_train_aug2.npy', mmap_mode='r')
Xt = np.load(f'{C}/cnn_test.npy', mmap_mode='r')
if HOLD >= 0:
    tri, hoi = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[HOLD]
else:
    tri, hoi = np.arange(len(y)), np.array([], int)
BS = 64
GAPS_TR = pd.read_csv(f'{D}/train_meta.csv').slew_gap_rows.values; GAPS_TE = pd.read_csv(f'{D}/test_meta.csv').slew_gap_rows.values
def tolast(a, gaps=None):
    a = np.asarray(a, np.float32)
    if not GLOBAL: return np.ascontiguousarray(a.transpose(0, 2, 3, 1))
    B = len(a); out = np.zeros((B, 128, 128, 4), np.float32)
    for g in np.unique(gaps):
        s = np.where(gaps == g)[0]
        for p in range(6):
            r0 = p * (16 + g)
            out[s, r0:r0 + 16, :, 0] = a[s, p]; out[s, r0:r0 + 16, :, 1] = a[s, 6 + p]
            out[s, r0:r0 + 16, :, 2] = 1.0 if p % 2 == 0 else -1.0; out[s, r0:r0 + 16, :, 3] = 1.0
    return out
net = Net2(4, 9, np.random.default_rng(SEED)) if GLOBAL else Net(12, 9, np.random.default_rng(SEED), width=WIDTH)
net.layers[-1].W *= 0.1
opt = Adam(net.params(), lr=1e-3, wd=1e-4)
CK = f'{C}/{TAG}_ckpt.pkl'; START = 0
if os.path.exists(CK):
    st = pickle.load(open(CK, 'rb'))
    for p, q in zip(net.params(), st['params']): p[...] = q
    for m, q in zip(opt.m, st['m']): m[...] = q
    for v, q in zip(opt.v, st['v']): v[...] = q
    opt.t = st['t']; START = st['ep'] + 1; print('resumed at epoch', START, flush=True)
rng = np.random.default_rng(SEED * 1000 + START)
def predict(X, idx, gaps):
    out = np.zeros((len(idx), 9), np.float32)
    for s in range(0, len(idx), 128):
        j = idx[s:s + 128]; logits = net.fwd(tolast(X[j], gaps[j]))
        z = logits - logits.max(1, keepdims=True); e = np.exp(z); out[s:s + len(j)] = e / e.sum(1, keepdims=True)
    return out
def save_preds():
    res = {}
    if len(hoi):
        res['hold_c'] = predict(Xc, hoi, GAPS_TR); res['hold_a'] = predict(Xa, hoi, GAPS_TR)
    res['test'] = predict(Xt, np.arange(len(Xt)), GAPS_TE)
    for k, a in res.items(): np.save(f'{C}/{TAG}_{k}.npy', a)
    return res
counts = np.bincount(y[tri], minlength=9); w = 1.0 / counts[y[tri]]; w /= w.sum()
PSEUDO = os.environ.get('PSEUDO')          # npz with test indices + labels (agreement pseudo-labels)
if PSEUDO:
    z = np.load(PSEUDO); pidx, plab = z['idx'], z['lab']
    pc_ = np.bincount(plab, minlength=9).clip(1); pw = 1.0 / pc_[plab]; pw /= pw.sum()
    print('pseudo test samples', len(pidx), flush=True)
steps = len(tri) // BS; t0 = time.time()
for ep in range(START, EPOCHS):
    opt.lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * ep / EPOCHS)) + 2e-5
    tl = 0
    for s in range(steps):
        b = np.sort(rng.choice(tri, BS, p=w))
        ua = rng.random(BS) < 0.5
        xb = np.where(ua[:, None, None, None], Xa[b], Xc[b]); yb = y[b]; gb = GAPS_TR[b]
        if PSEUDO:
            npse = BS // 4; pj = np.sort(rng.choice(len(pidx), npse, p=pw))
            xb = np.concatenate([xb[:BS - npse], Xt[pidx[pj]]]); yb = np.concatenate([yb[:BS - npse], plab[pj]])
            gb = np.concatenate([gb[:BS - npse], GAPS_TE[pidx[pj]]])
        xb = tolast(xb, gb)
        sh = rng.integers(-6, 7)
        if sh: xb = np.roll(xb, sh, axis=2)
        L, d, _ = softmax_ce(net.fwd(xb), yb); net.bwd(d); opt.step(net.grads()); tl += L
    pickle.dump({'params': net.params(), 'm': opt.m, 'v': opt.v, 't': opt.t, 'ep': ep}, open(CK + '.tmp', 'wb'))
    os.replace(CK + '.tmp', CK)
    msg = f'ep {ep} lr {opt.lr:.5f} loss {tl / steps:.4f} t {time.time() - t0:.0f}s'
    if ep % 3 == 2 or ep >= EPOCHS - 2:
        r = save_preds()
        if len(hoi):
            msg += f" | hold clean {f1_score(y[hoi], r['hold_c'].argmax(1), average='macro'):.4f} aug {f1_score(y[hoi], r['hold_a'].argmax(1), average='macro'):.4f}"
    print(msg, flush=True)
print('done', flush=True)
