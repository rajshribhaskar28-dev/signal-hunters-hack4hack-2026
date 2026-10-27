"""Preprocessing shared by all feature extractors.

Input: uint8 cadences (B, 6, 16, 128). 0 = missing.
Output: z (B,6,16,128) float32 robust-standardised residual with NaN for invalid pixels,
        valid mask, glitch-row mask.
Steps (all per cadence, per panel, so they are telescope agnostic):
  1. values -> float, 0 -> NaN; channel 64 (GBT centre spike) -> NaN for every telescope.
  2. Broadband glitch rows: row median of (v - per-channel time median) far above the
     robust noise -> mask whole row.
  3. Bandpass + panel gain: per panel per channel time-median, smoothed along frequency
     with a 9-channel running median (so narrow lines survive), subtracted in log domain
     (gain = additive offset in log domain).
  4. Noise: per panel MAD -> z-score. Makes PKS (noisier) and GBT comparable.
"""
import numpy as np
from scipy.ndimage import median_filter

SPIKE_CH = 64


def _nanmad(x, axis):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(x - med), axis=axis, keepdims=True) * 1.4826


def preprocess(u8, glitch_k=0.6, smooth=9):
    v = u8.astype(np.float32)
    v[v == 0] = np.nan
    v[..., SPIKE_CH] = np.nan
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med1 = np.nanmedian(v, axis=2, keepdims=True)                 # B,6,1,128
            r1 = v - med1
            s1 = _nanmad(r1, axis=(2, 3))                                  # B,6,1,1
            rm = np.nanmedian(r1, axis=3, keepdims=True)                   # B,6,16,1
            glitch = (np.abs(rm) > glitch_k * s1)                          # B,6,16,1
            v = np.where(glitch, np.nan, v)
            med = np.nanmedian(v, axis=2)                                  # B,6,128
            # fill NaN channels (spike / fully missing) with panel median before filtering
            pm = np.nanmedian(med, axis=2, keepdims=True)
            pm = np.where(np.isnan(pm), 96.0, pm)
            med = np.where(np.isnan(med), pm, med)
            bp = median_filter(med, size=(1, 1, smooth), mode='nearest')   # B,6,128
            r = v - bp[:, :, None, :]
            s = _nanmad(r, axis=(2, 3))
            s = np.where((s > 0) & np.isfinite(s), s, 1.0)
            z = (r - np.nanmedian(r, axis=(2, 3), keepdims=True)) / s
            # tail equalisation: match the upper noise quantile (q90) across receivers
            q90 = np.nanquantile(z, 0.9, axis=(2, 3), keepdims=True)
            up = np.where(np.isfinite(q90) & (q90 > 0.3), q90 / 1.2816, 1.0)
            z = z / up
    valid = np.isfinite(z)
    return z.astype(np.float32), valid, glitch[..., 0], s[:, :, 0, 0]
