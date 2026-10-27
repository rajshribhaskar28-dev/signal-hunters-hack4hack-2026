"""Precompute CNN inputs: preprocess.py z-scores, NaN->0, clipped to [-4,10], float16, NHWC (N,16,128,6)."""
import sys, numpy as np, time
from prep import preprocess
from features import D
from augment import pks_like
split = sys.argv[1]; aug = int(sys.argv[2]) if len(sys.argv) > 2 else None
X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
out = np.zeros((len(X), 16, 128, 6), np.float16); t = time.time()
for s in range(0, len(X), 200):
    u8 = np.asarray(X[s:s + 200])
    if aug is not None: u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
    z, v, _, _ = preprocess(u8)
    z = np.clip(np.where(v, z, 0), -4, 10)
    out[s:s + 200] = z.transpose(0, 2, 3, 1)
tag = split if aug is None else f'{split}_aug{aug}'
np.save(f'cache/cnn_{tag}.npy', out); print('done', tag, out.shape, round(time.time() - t), flush=True)
