"""Feature group 4: line-robust broad-emission features.

Narrow RFI lines inflate smoothed-texture statistics, hiding faint scintillation. Here the
cross-panel-bandpass residual z (as in features2) is clipped to [-2.5, 2.5] before smoothing,
and an ON-mean minus OFF-mean difference map cancels anything present in both ON and OFF
panels (persistent RFI, bandpass, radar), leaving ON-only emission.
python features4.py <train|test> [aug_seed]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from features import panel_aggs, D, ON, OFF

warnings.filterwarnings('ignore')


def resid(u8):
    v = u8.astype(np.float32); v[v == 0] = np.nan; v[..., 64] = np.nan
    med1 = np.nanmedian(v, axis=2, keepdims=True); r1 = v - med1
    s1 = np.nanmedian(np.abs(r1), axis=(2, 3), keepdims=True) * 1.4826
    rm = np.nanmedian(r1, axis=3, keepdims=True)
    v = np.where(np.abs(rm) > 0.6 * s1, np.nan, v)
    v = v - np.nanmedian(v, axis=(2, 3), keepdims=True)
    tm = np.nan_to_num(np.nanmedian(v, axis=2))
    # common bandpass from the OFF panels only (used for every panel)
    bp = np.median(tm[:, OFF], axis=1)                                 # B,128
    r = v - bp[:, None, None, :]
    s = np.where(s1[:, :, 0, 0] > 0, s1[:, :, 0, 0], 1)
    z = r / s[:, :, None, None]
    return z, np.isfinite(z)


def extract4(u8):
    B = u8.shape[0]
    z, valid = resid(u8)
    zc = np.clip(np.where(valid, z, 0), -2.5, 2.5)
    out = {}
    for (ft, ff) in [(4, 16), (8, 32), (16, 64), (16, 16)]:
        cnt = uniform_filter(valid.astype(np.float32), size=(1, 1, ft, ff), mode='constant')
        sm = np.where(cnt > 0.3, uniform_filter(zc, size=(1, 1, ft, ff), mode='constant') / np.maximum(cnt, 1e-3), 0)
        panel_aggs(f'c_sm{ft}x{ff}_std', sm.std((2, 3)), out)
        panel_aggs(f'c_sm{ft}x{ff}_q95', np.quantile(sm.reshape(B, 6, -1), 0.95, axis=-1), out)
        # ON-minus-OFF difference map (same smoothing)
        dmap = sm[:, ON].mean(1) - sm[:, OFF].mean(1)                     # B,16,128
        out[f'c_diff{ft}x{ff}_max'] = dmap.max((1, 2))
        out[f'c_diff{ft}x{ff}_q90'] = np.quantile(dmap.reshape(B, -1), 0.9, axis=-1)
        out[f'c_diff{ft}x{ff}_mean'] = dmap.mean((1, 2))
        out[f'c_diff{ft}x{ff}_std'] = dmap.std((1, 2))
    panel_aggs('c_mean', zc.mean((2, 3)), out)
    panel_aggs('c_frac_pos1', (zc > 1).mean((2, 3)), out)
    # fraction of channels with ON-only excess (patchy multichannel brightening)
    spec = zc.mean(2)                                                     # B,6,128
    dspec = spec[:, ON].mean(1) - spec[:, OFF].mean(1)
    out['c_dspec_frac03'] = (dspec > 0.3).mean(1)
    out['c_dspec_mean'] = dspec.mean(1)
    out['c_dspec_std8'] = uniform_filter(dspec, size=(1, 8), mode='nearest').std(1)
    # matched 2-D power-spectrum detector: scintillation excess at k_t=1..2, k_f=0..4
    from prep import preprocess
    zp, vp, _, _ = preprocess(u8)
    zp = np.where(vp, zp, 0); zp = zp - zp.mean((2, 3), keepdims=True)
    P = np.abs(np.fft.rfft2(zp, axes=(2, 3))) ** 2                          # B,6,16,65
    noise = P[:, :, 4:13, 8:].mean((2, 3)) + 1e-6
    for nm, (a, b, c, d) in {'k12_04': (1, 3, 0, 5), 'k1_02': (1, 2, 0, 3), 'k12_14': (1, 3, 1, 5),
                             'k3_04': (3, 4, 0, 5), 'k0_14': (0, 1, 1, 5)}.items():
        bp = (P[:, :, a:b, c:d].mean((2, 3)) + P[:, :, 16 - b + 1:16 - a + 1, c:d].mean((2, 3))) / 2 if a > 0 else P[:, :, a:b, c:d].mean((2, 3))
        panel_aggs(f'ps_{nm}', np.log(bp / noise), out)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract4(u8))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat4_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
