"""Line maps for MapNet: (N, 144, 42) float16. Per panel 7 maps over channel position (pad 8):
stationary, +slow, -slow, signed drift of best slow line (where strong), fast, spectrum mean, |slow best|."""
import sys, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from prep import preprocess
from drift import drift_sums
from features import SLOW, FAST, PAD, D
from augment import pks_like
split = sys.argv[1]; aug = int(sys.argv[2]) if len(sys.argv) > 2 else None
X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
zero = np.abs(SLOW) <= 0.0625; pos = SLOW > 0.0625; neg = SLOW < -0.0625
out = np.zeros((len(X), 128 + 2 * PAD, 42), np.float16); t0 = time.time()
for s in range(0, len(X), 200):
    u8 = np.asarray(X[s:s + 200])
    if aug is not None: u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
    z, v, _, _ = preprocess(u8)
    S, _ = drift_sums(z, v, SLOW, PAD); Sf, _ = drift_sums(z, v, FAST, PAD)
    best = S.max(2); dsg = SLOW[S.argmax(2)] * (best > 4)
    spec = np.zeros((len(u8), 6, 128 + 2 * PAD), np.float32)
    zf = np.where(v, z, 0); spec[:, :, PAD:PAD + 128] = zf.sum(2) / np.sqrt(np.maximum(v.sum(2), 1))
    maps = np.stack([S[:, :, zero].max(2), S[:, :, pos].max(2), S[:, :, neg].max(2), dsg * 5,
                     Sf.max(2), spec, best], -1)                       # B,6,W,7
    maps = np.clip(maps, -3, 20) / 5.0
    out[s:s + 200] = maps.transpose(0, 2, 1, 3).reshape(len(u8), 128 + 2 * PAD, 42)
tag = split if aug is None else f'{split}_aug{aug}'
np.save(f'cache/maps_{tag}.npy', out); print('done', tag, out.shape, round(time.time() - t0), flush=True)
