"""PKS-like perturbation of GBT training cadences (domain-shift stress test / augmentation).
Parameters were estimated only from unlabeled test statistics (noise MAD, edge roll-off,
glitch/dropped-row rates, panel gain spread), never from test labels."""
import numpy as np


K_RANGE = {2: (4.0, 8.0), 3: (4.0, 8.0)}   # original (too strong) copies; others use the test-matched range


def pks_like(u8, rng, seed=None):
    B = u8.shape[0]
    v = u8.astype(np.float32)
    miss = v == 0
    # remove GBT centre spike by neighbour interpolation
    v[..., 64] = 0.5 * (v[..., 63] + v[..., 65])
    ch = np.arange(128) / 127.0
    # noisier receiver: GBT log-noise ~ Gamma(k~8), PKS ~ Gamma(k~4.6)
    # fewer averaged degrees of freedom: multiply linear power by Gamma(k,1/k) noise
    lo, hi = K_RANGE.get(seed, (8.0, 16.0))   # (8,16) matches PKS pixel autocorrelation (0.042 vs 0.046)
    k = rng.uniform(lo, hi, size=(B, 1, 1, 1))
    p = np.exp2(v / 24.0 - 4.0) * rng.gamma(k, 1.0 / k, size=v.shape)
    v = (24.0 * (np.log2(np.maximum(p, 1e-12)) + 4.0)).astype(np.float32)
    # bandpass: roll-off, tilt, ripple
    roll_lo = rng.uniform(10, 40, (B, 1, 1, 1)); roll_hi = rng.uniform(5, 30, (B, 1, 1, 1))
    w = rng.uniform(0.04, 0.12, (B, 1, 1, 1))
    bp = -roll_lo * np.exp(-ch / w) - roll_hi * np.exp(-(1 - ch) / w)
    bp += rng.uniform(-6, 6, (B, 1, 1, 1)) * (ch - 0.5)
    per = rng.uniform(10, 40, (B, 1, 1, 1)); ph = rng.uniform(0, 2 * np.pi, (B, 1, 1, 1))
    bp += rng.uniform(0, 4, (B, 1, 1, 1)) * np.sin(2 * np.pi * np.arange(128) / per + ph)
    v += bp.astype(np.float32)
    # panel gain changes
    v += rng.normal(0, 2.5, (B, 6, 1, 1)).astype(np.float32)
    # broadband glitch rows
    g = rng.random((B, 6, 16, 1)) < 0.009
    v += np.where(g, rng.uniform(15, 60, (B, 6, 16, 1)), 0).astype(np.float32)
    # dropped rows
    d = rng.random((B, 6, 16, 1)) < 0.0025
    out = np.clip(np.round(v), 1, 255).astype(np.uint8)
    out[miss | np.broadcast_to(d, out.shape)] = 0
    return out
