"""PKS-like augmentation of GBT training cadences in the raw uint8 domain.
Parameters are estimated from unlabeled test statistics only (bandpass median profile, panel gain spread,
log-domain noise width, dropped/glitch row rates) — no test labels are involved."""
import numpy as np

def bandpass_profiles(X, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), min(n, len(X)), replace=False))
    a = np.asarray(X[idx]).astype(np.float32); a[a == 0] = np.nan
    a[..., 64] = 0.5 * (a[..., 63] + a[..., 65])
    prof = np.nanmedian(a, axis=(0, 1, 2))
    return prof - np.nanmedian(prof)

def pksify(v, rng, bp_delta, gain_sd=1.9, noise_k=0.5, noise_sd=11.5, p_drop=0.0025, p_glitch=0.009,
           tilt_sd=3.0, strength=1.0):
    """v: (B,6,16,128) uint8 -> augmented uint8."""
    B = v.shape[0]
    x = v.astype(np.float32)
    drop0 = (v == 0)
    x[..., 64] = 0.5 * (x[..., 63] + x[..., 65])           # no centre spike on PKS
    # bandpass shape change (+ random tilt and ripple shift)
    scale = rng.uniform(0.7, 1.3, size=(B, 1))
    bp = scale * bp_delta[None, :]                          # (B,128)
    tilt = rng.normal(0, tilt_sd, size=(B, 1)) * np.linspace(-1, 1, 128)[None]
    x += strength * (bp + tilt)[:, None, None, :]
    # larger panel gain changes
    x += strength * rng.normal(0, gain_sd, size=(B, 6, 1, 1))
    # extra multiplicative (chi-square like) noise: power *= Gamma(k)/k
    if noise_k:
        # log-domain noise with PKS-like left skew; shape/scale fitted to unlabeled test residual quantiles & z-tails
        n = 24.0 * np.log2(rng.gamma(noise_k, 1.0, size=x.shape).astype(np.float32) + 1e-30)
        n = np.clip(n, -400, None)
        x += (n - n.mean()) / (n.std() + 1e-6) * noise_sd
    # extra broadband glitch rows
    gl = rng.random((B, 6, 16)) < p_glitch
    amp = rng.uniform(10, 50, size=gl.sum())
    sign = np.where(rng.random(gl.sum()) < 0.15, -0.4, 1.0)
    x[gl] += (amp * sign)[:, None]
    x = np.clip(np.round(x), 1, 255)
    # extra dropped rows
    dr = rng.random((B, 6, 16)) < p_drop
    x[dr] = 0
    x[drop0] = 0
    return x.astype(np.uint8)
