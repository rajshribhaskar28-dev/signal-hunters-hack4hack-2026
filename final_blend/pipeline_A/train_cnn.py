"""Compact 2D CNN over all 6 panels jointly (tinygrad, CPU). Trained on folds != HOLD of the shared
StratifiedKFold(5, seed 42); each training cadence is shown either clean or PKS-augmented (aug seed 2)."""
import sys, time, os, numpy as np, pandas as pd
sys.path.insert(0, os.environ.get('TINYGRAD_PATH', 'tinygrad')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, nn, TinyJit, Context, dtypes
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
SEED = int(sys.argv[1]); EPOCHS = int(sys.argv[2]); HOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 0
D = os.environ.get('SH_DATA', 'data')
CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
           'rfi_intermittent', 'rfi_persistent', 'noise']
tr = pd.read_csv(f'{D}/train_meta.csv'); y = tr.label.map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int32)
Xc = np.load('cache/cnn_train.npy', mmap_mode='r')
Xa = np.load('cache/cnn_train_aug2.npy', mmap_mode='r')
Xt = np.load('cache/cnn_test.npy', mmap_mode='r')
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
tri, hoi = folds[HOLD]
Tensor.manual_seed(SEED); rng = np.random.default_rng(SEED)

class Net:
    def __init__(self, cin=12):
        self.c1 = nn.Conv2d(cin, 24, (3, 5), padding=(1, 2)); self.b1 = nn.BatchNorm(24)
        self.c2 = nn.Conv2d(24, 32, (3, 5), padding=(1, 2)); self.b2 = nn.BatchNorm(32)
        self.c3 = nn.Conv2d(32, 48, 3, padding=1); self.b3 = nn.BatchNorm(48)
        self.l1 = nn.Linear(96, 64); self.l2 = nn.Linear(64, 9)
    def __call__(self, x):
        x = self.b1(self.c1(x)).relu().max_pool2d((1, 2))
        x = self.b2(self.c2(x)).relu().max_pool2d((2, 2))
        x = self.b3(self.c3(x)).relu()
        h = x.max(axis=(2, 3)).cat(x.mean(axis=(2, 3)), dim=1)
        return self.l2(self.l1(h).relu().dropout(0.2))

net = Net(); params = nn.state.get_parameters(net)
opt = nn.optim.Adam(params, lr=2e-3)
BS = 128

@TinyJit
@Context(TRAINING=1)
def step(x, yb):
    opt.zero_grad(); loss = net(x).sparse_categorical_crossentropy(yb).backward()
    return loss.realize(*opt.schedule_step())

@TinyJit
def infer(x): return net(x).softmax().realize()

def predict(X, idx):
    out = np.zeros((len(idx), 9), np.float32)
    for s in range(0, len(idx), BS):
        j = idx[s:s + BS]; xb = np.asarray(X[np.sort(j)] if False else X[j]).astype(np.float32)
        n = len(j)
        if n < BS: xb = np.concatenate([xb, np.zeros((BS - n,) + xb.shape[1:], np.float32)])
        out[s:s + n] = infer(Tensor(xb)).numpy()[:n]
    return out

counts = np.bincount(y[tri], minlength=9); w = 1.0 / counts[y[tri]]; w /= w.sum()
steps = len(tri) // BS
t0 = time.time()
for ep in range(EPOCHS):
    lr = 2e-3 * (0.5 * (1 + np.cos(np.pi * ep / EPOCHS))) + 1e-4
    opt.lr.assign(Tensor([lr], device=opt.lr.device, dtype=opt.lr.dtype)).realize()
    tl = 0
    for s in range(steps):
        b = np.sort(rng.choice(tri, BS, p=w))
        use_aug = rng.random(BS) < 0.5
        xb = np.where(use_aug[:, None, None, None], Xa[b], Xc[b]).astype(np.float32)
        sh = rng.integers(-6, 7)                      # small frequency translation
        if sh: xb = np.roll(xb, sh, axis=-1)
        tl += step(Tensor(xb), Tensor(y[b])).item()
    msg = f'ep {ep} lr {lr:.4f} loss {tl / steps:.4f} t {time.time() - t0:.0f}s'
    if ep >= EPOCHS - 3 or ep % 4 == 3:
        with Context(TRAINING=0):
            pc = predict(Xc, hoi); pa = predict(Xa, hoi)
        msg += f' | hold clean {f1_score(y[hoi], pc.argmax(1), average="macro"):.4f} aug {f1_score(y[hoi], pa.argmax(1), average="macro"):.4f}'
    print(msg, flush=True)
with Context(TRAINING=0):
    pc = predict(Xc, hoi); pa = predict(Xa, hoi); pt = predict(Xt, np.arange(len(Xt)))
np.save(f'cache/cnn_s{SEED}_h{HOLD}_hold_c.npy', pc); np.save(f'cache/cnn_s{SEED}_h{HOLD}_hold_a.npy', pa)
np.save(f'cache/cnn_s{SEED}_h{HOLD}_test.npy', pt)
print('done', time.time() - t0, flush=True)
