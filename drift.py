"""Straight-line drift search (de-Doppler style) on standardised spectrogram panels."""
import numpy as np


def drift_sums(z, valid, drifts, pad, min_rows=4):
    """S[b,p,k,j] = sum_r z[b,p,r, c_j + round(d_k*(r-7.5)+0.5*...)]/sqrt(n)
    Track is parameterised by its channel at the panel mid-time (r=7.5): c_mid = j - pad.
    Returns S (B,6,ND,128+2pad) float32 and N (valid counts)."""
    B, P, R, C = z.shape
    zz = np.where(valid, z, 0).astype(np.float32)
    vv = valid.astype(np.float32)
    W = C + 2 * pad
    big = pad + 300
    zp = np.zeros((B, P, R, C + 2 * big), np.float32)
    vp = np.zeros_like(zp)
    zp[..., big:big + C] = zz
    vp[..., big:big + C] = vv
    S = np.zeros((B, P, len(drifts), W), np.float32)
    N = np.zeros((B, P, len(drifts), W), np.float32)
    t = np.arange(R) - (R - 1) / 2.0
    for k, d in enumerate(drifts):
        off = np.floor(d * t + 0.5).astype(int)
        acc = np.zeros((B, P, W), np.float32)
        cnt = np.zeros((B, P, W), np.float32)
        for r in range(R):
            s0 = big - pad + off[r]
            acc += zp[:, :, r, s0:s0 + W]
            cnt += vp[:, :, r, s0:s0 + W]
        S[:, :, k] = np.where(cnt >= min_rows, acc / np.sqrt(np.maximum(cnt, 1)), 0)
        N[:, :, k] = cnt
    return S, N
