"""GPU CNN + blend with the v9 gradient-boosting model (run on Colab / Kaggle with a GPU).

Usage (Colab, GPU runtime):
    !git clone https://github.com/rajshribhaskar28-dev/signal-hunters-hack4hack-2026 sh
    %cd sh
    !unzip -q /content/signal-hunters-round2.zip -d /content/data
    !python gpu_cnn.py --data /content/data/signal-hunters-round2

What it does
 1. Builds CNN inputs with the same preprocessing as the feature model (prep.py: missing-data
    masking, glitch rows, bandpass/gain flattening, robust z, noise-tail equalisation) for clean
    GBT train, a harsh (aug 2) and a test-matched (aug 4) PKS-like copy, and the PKS test set.
 2. Trains a compact residual-free VGG-style CNN on fold-0 train rows (same split as the GBM CV),
    reports macro-F1 on fold-0 validation rows, clean and test-matched.
 3. Chooses the CNN/GBM blend weight on those validation rows using the GBM fold-0 probabilities
    shipped in blend/, retrains the CNN on all rows, predicts test, blends with v9 test
    probabilities and writes submission_cnn_blend.csv (+ submission_cnn_only.csv).
"""
import argparse, os, sys, time, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
           'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']
ON, OFF = [0, 2, 4], [1, 3, 5]

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--cache', default='/content/cnn_cache' if os.path.isdir('/content') else 'cnn_cache')
ap.add_argument('--skip-full', action='store_true', help='skip the all-rows retrain (use fold-0 model for test)')
args = ap.parse_args()
os.environ['SH_DATA'] = args.data
os.makedirs(args.cache, exist_ok=True)

from prep import preprocess
from augment import pks_like


def build(split, aug=None):
    path = f'{args.cache}/{split}_{aug}.npy'
    if os.path.exists(path):
        return np.load(path)
    X = np.load(f'{args.data}/{split}_cadences.npy', mmap_mode='r')
    out = np.zeros((len(X), 6, 16, 128), np.float16); t = time.time()
    for s in range(0, len(X), 200):
        u8 = np.asarray(X[s:s + 200])
        if aug is not None:
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        z, v, _, _ = preprocess(u8)
        out[s:s + 200] = np.clip(np.where(v, z, 0), -4, 10)
    np.save(path, out); print(f'built {split} aug={aug} {out.shape} in {time.time() - t:.0f}s', flush=True)
    return out


print('building inputs (CPU, ~3 min each, cached)...', flush=True)
Xc, Xh, Xm, Xt = build('train'), build('train', 2), build('train', 4), build('test')
tr = pd.read_csv(f'{args.data}/train_meta.csv'); y = np.array([CLASSES.index(v) for v in tr.label])

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev, flush=True)
torch.manual_seed(0); np.random.seed(0)


def add_channels(x):
    """x: (B,6,16,128) -> (B,9,16,128): panels + ON-mean minus OFF-mean, ON-min, OFF-max."""
    on, off = x[:, ON], x[:, OFF]
    return torch.cat([x, (on.mean(1) - off.mean(1)).unsqueeze(1), on.min(1)[0].unsqueeze(1),
                      off.max(1)[0].unsqueeze(1)], 1)


def cbr(ci, co, k=(3, 3)):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=(k[0] // 2, k[1] // 2), bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, w=64, ncls=9):
        super().__init__()
        self.f = nn.Sequential(
            cbr(9, w, (3, 5)), cbr(w, w), nn.MaxPool2d(2),                 # 8 x 64
            cbr(w, 2 * w), cbr(2 * w, 2 * w), nn.MaxPool2d(2),            # 4 x 32
            cbr(2 * w, 4 * w), cbr(4 * w, 4 * w), nn.MaxPool2d((1, 2)),   # 4 x 16
            cbr(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * w, ncls))

    def forward(self, x):
        h = self.f(add_channels(x))
        return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))


SRC = [Xc, Xm, Xh]                     # clean, test-matched, harsh
PSRC = np.array([0.4, 0.35, 0.25])


def batch(ids, rng):
    src = rng.choice(3, len(ids), p=PSRC)
    x = np.empty((len(ids), 6, 16, 128), np.float32)
    for k in range(3):
        m = src == k
        if m.any(): x[m] = SRC[k][ids[m]]
    sh = int(rng.integers(-12, 13))                     # small frequency translation, zero fill
    if sh:
        x = np.roll(x, sh, axis=3)
        if sh > 0: x[..., :sh] = 0
        else: x[..., sh:] = 0
    x *= rng.uniform(0.9, 1.1, (len(ids), 1, 1, 1)).astype(np.float32)   # small gain jitter
    return torch.from_numpy(x).to(dev), torch.from_numpy(y[ids]).to(dev)


@torch.no_grad()
def predict(model, X, idx=None, bs=1024):
    model.eval(); idx = np.arange(len(X)) if idx is None else idx; out = []
    for s in range(0, len(idx), bs):
        xb = torch.from_numpy(X[idx[s:s + bs]].astype(np.float32)).to(dev)
        with torch.autocast(device_type='cuda', enabled=dev == 'cuda'):
            out.append(F.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out)


def train(rows, epochs, val=None, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = Net(args.width).to(dev)
    cnt = np.bincount(y[rows], minlength=9); cw = torch.tensor(len(rows) / (9 * cnt), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    steps = int(np.ceil(2 * len(rows) / args.bs))      # 2 passes' worth of mixed-source samples per epoch
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    hist = []
    for ep in range(epochs):
        model.train(); t = time.time(); L = 0
        for _ in range(steps):
            xb, yb = batch(rng.choice(rows, args.bs), rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=dev == 'cuda'):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); L += loss.item()
        msg = f'ep {ep:2d} loss {L / steps:.4f} {time.time() - t:.0f}s'
        if val is not None and (ep % 5 == 4 or ep == epochs - 1):
            Pc, Pm = predict(model, Xc, val), predict(model, Xm, val)
            fc, fm = f1_score(y[val], Pc.argmax(1), average='macro'), f1_score(y[val], Pm.argmax(1), average='macro')
            hist.append((ep, fc, fm)); msg += f' | val clean {fc:.4f} matched {fm:.4f}'
        print(msg, flush=True)
    return model, hist


tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))[0]
print(f'== fold-0 training ({len(tri)} rows), validation {len(vai)} rows', flush=True)
model, hist = train(tri, args.epochs, val=vai)
Pc, Pm = predict(model, Xc, vai), predict(model, Xm, vai)

# ---- blend weight on fold-0 validation (GBM probabilities for the same rows are in blend/) ----
Gc = np.load(f'{HERE}/blend/P_gbm_cleanval_f0.npy'); Gm = np.load(f'{HERE}/blend/P_gbm_matchedval_f0.npy')
assert (np.load(f'{HERE}/blend/va_f0.npy') == vai).all(), 'validation rows differ from the GBM fold-0 rows'
print('\nblend weight w (CNN share): clean / test-matched macro-F1 on fold-0 validation')
best = (-1, 0)
for w in [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    fc = f1_score(y[vai], ((1 - w) * Gc + w * Pc).argmax(1), average='macro')
    fm = f1_score(y[vai], ((1 - w) * Gm + w * Pm).argmax(1), average='macro')
    print(f'  w={w:.1f}: clean {fc:.4f}  matched {fm:.4f}')
    if fm + 0.5 * fc > best[0]: best = (fm + 0.5 * fc, w)
W = best[1]; print(f'chosen CNN weight {W}', flush=True)

# ---- final CNN on all rows, predict test ----
if not args.skip_full:
    print('== retraining CNN on all rows', flush=True)
    model, _ = train(np.arange(len(y)), args.epochs, seed=1)
Pt = predict(model, Xt)
np.save('testP_cnn.npy', Pt)
Gt = np.load(f'{HERE}/blend/testP_gbm_v9.npy')
te = pd.read_csv(f'{args.data}/test_meta.csv'); ss = pd.read_csv(f'{args.data}/sample_submission.csv')
assert (te.id.values == ss.id.values).all()
for name, P in [('submission_cnn_blend.csv', (1 - W) * Gt + W * Pt), ('submission_cnn_only.csv', Pt)]:
    sub = pd.DataFrame({'id': ss.id, 'label': [CLASSES[i] for i in P.argmax(1)]})
    assert len(sub) == 10000 and sub.id.is_unique and sub.label.isin(CLASSES).all()
    sub.to_csv(name, index=False); print('wrote', name, sub.label.value_counts().to_dict())
json.dump({'blend_weight': W, 'fold0_hist': hist}, open('cnn_run.json', 'w'))
print('\nUPLOAD submission_cnn_blend.csv if the chosen blend beats w=0 on the matched column above.')
