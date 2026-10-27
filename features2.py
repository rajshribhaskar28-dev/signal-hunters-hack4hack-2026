"""Feature group 2: broad-structure / scintillation features with a cross-panel bandpass.

The per-panel smoothed bandpass in prep.py absorbs structure wider than ~9 channels that
persists in time, which hides scintillation patches. Here each panel's bandpass is the
median over the OTHER five panels' time-median spectra (leave-one-panel-out), after
removing each panel's scalar gain (additive in the log domain). Broad brightening that
is present only in ON panels therefore survives.
python features2.py <train|test> [aug_seed]
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from features import panel_aggs, D, ON, OFF

warnings.filterwarnings('ignore')


def extract2(u8):
    B = u8.shape[0]
    v = u8.astype(np.float32)
    v[v == 0] = np.nan
    v[..., 64] = np.nan
    # glitch rows (same rule as prep)
    med1 = np.nanmedian(v, axis=2, keepdims=True)
    r1 = v - med1
    s1 = np.nanmedian(np.abs(r1 - np.nanmedian(r1, axis=(2, 3), keepdims=True)), axis=(2, 3), keepdims=True) * 1.4826
    rm = np.nanmedian(r1, axis=3, keepdims=True)
    v = np.where(np.abs(rm) > 0.6 * s1, np.nan, v)
    gain = np.nanmedian(v, axis=(2, 3), keepdims=True)                  # B,6,1,1
    v = v - gain
    tm = np.nanmedian(v, axis=2)                                         # B,6,128 time-median spectrum
    tm = np.where(np.isnan(tm), 0, tm)
    loo = np.stack([np.median(np.delete(tm, p, axis=1), axis=1) for p in range(6)], 1)   # B,6,128
    r = v - loo[:, :, None, :]
    sig = np.nanmedian(np.abs(r1), axis=(2, 3)) * 1.4826                # B,6 per-pixel noise
    sig = np.where(sig > 0, sig, 1)
    z = r / sig[:, :, None, None]
    valid = np.isfinite(z)
    zf = np.where(valid, z, 0)
    out = {}
    spec = (zf.sum(2) / np.maximum(valid.sum(2), 1))                    # B,6,128 mean excess per channel
    spec = spec - np.median(spec, axis=2, keepdims=True)
    for w in [1, 4, 16]:
        s = uniform_filter(spec, size=(1, 1, w), mode='nearest') if w > 1 else spec
        panel_aggs(f'x_spec{w}_std', s.std(2), out)
        panel_aggs(f'x_spec{w}_max', s.max(2), out)
    for (ft, ff) in [(4, 8), (8, 16), (16, 32), (4, 32)]:
        cnt = uniform_filter(valid.astype(np.float32), size=(1, 1, ft, ff), mode='constant')
        sm = uniform_filter(zf, size=(1, 1, ft, ff), mode='constant') / np.maximum(cnt, 1e-3)
        sm = np.where(cnt > 0.3, sm, 0)
        sm = sm - np.median(sm, axis=(2, 3), keepdims=True)
        panel_aggs(f'x_sm{ft}x{ff}_std', sm.std((2, 3)), out)
        panel_aggs(f'x_sm{ft}x{ff}_max', sm.max((2, 3)), out)
        panel_aggs(f'x_sm{ft}x{ff}_q90', np.quantile(sm.reshape(B, 6, -1), 0.9, axis=-1), out)
    # temporal variability of channel-averaged power (twinkling in time)
    ts = zf.sum(3) / np.maximum(valid.sum(3), 1)                         # B,6,16
    panel_aggs('x_ts_std', ts.std(2), out)
    # time-frequency texture: correlation between adjacent channels and adjacent rows
    a = zf - zf.mean((2, 3), keepdims=True)
    den = (a * a).mean((2, 3)) + 1e-6
    panel_aggs('x_ac_f', (a[..., 1:] * a[..., :-1]).mean((2, 3)) / den, out)
    panel_aggs('x_ac_t', (a[:, :, 1:] * a[:, :, :-1]).mean((2, 3)) / den, out)
    panel_aggs('x_ac_f4', (a[..., 4:] * a[..., :-4]).mean((2, 3)) / den, out)
    # gain step ON vs OFF (scalar brightness), in noise units
    g = gain[:, :, 0, 0] / sig
    out['x_gain_on_m_off'] = (g[:, ON].mean(1) - g[:, OFF].mean(1))
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s))
        parts.append(extract2(u8))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat2_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
