"""Feature group 6: precise multi-panel line geometry + raw ON-brightening level.

Persistent RFI = one transmitter: the same channel AND drift in (almost) every panel.
Intermittent RFI often also has a line in every panel, but at different channels / drifts
(several transmitters). For each panel the two strongest slow-drift lines (channel at panel
mid-time, drift, significance) are found; the 12 detections are grouped across panels when
|dc| <= 1.5 ch and |dd| <= 0.07 ch/row. Cluster statistics describe the cadence.
Raw level: scintillating stars brighten the ON panels as a whole (+~4 units vs 0 for noise),
which per-panel gain normalisation removes; measured here from the raw bytes.
python features6.py <train|test> [aug_seed]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from prep import preprocess
from drift import drift_sums
from features import SLOW, PAD, D, ON, OFF

warnings.filterwarnings('ignore')
NB = 2


def top_lines(S):
    """S (B,6,ND,W) -> arrays (B,6,NB) of score, channel(mid), drift for the NB strongest
    separated lines per panel (separation >= 4 channels)."""
    B, P, ND, W = S.shape
    prof = S.max(2); kd = S.argmax(2)                      # B,6,W
    sc = np.zeros((B, P, NB), np.float32); ch = np.zeros((B, P, NB)); dr = np.zeros((B, P, NB))
    work = prof.copy()
    for n in range(NB):
        j = work.argmax(-1)                                # B,6
        sc[..., n] = np.take_along_axis(work, j[..., None], -1)[..., 0]
        ch[..., n] = j - PAD
        dr[..., n] = SLOW[np.take_along_axis(kd, j[..., None], -1)[..., 0]]
        lo = np.clip(j - 4, 0, W); hi = np.clip(j + 5, 0, W)
        ar = np.arange(W)[None, None, :]
        work = np.where((ar >= lo[..., None]) & (ar < hi[..., None]), -1e9, work)
    return sc, ch, dr


def geometry(sc, ch, dr, thr):
    B = sc.shape[0]
    pan = np.repeat(np.arange(6), NB)[None, :].repeat(B, 0)             # B,12
    s = sc.reshape(B, -1); c = ch.reshape(B, -1); d = dr.reshape(B, -1)
    lit = s > thr
    same = (np.abs(c[:, :, None] - c[:, None, :]) <= 1.5) & (np.abs(d[:, :, None] - d[:, None, :]) <= 0.07)
    same &= lit[:, :, None] & lit[:, None, :]
    # panels covered by the cluster seeded at each detection
    pm = np.zeros((B, 12, 6), bool)
    for p in range(6):
        pm[:, :, p] = (same & (pan[:, None, :] == p)).any(-1)
    cover = pm.sum(-1)                                                   # B,12
    on_cov = pm[:, :, ON].sum(-1); off_cov = pm[:, :, OFF].sum(-1)
    out = {}
    best = np.where(lit, cover, 0).argmax(1)
    bi = np.arange(B)
    out['big_cover'] = cover[bi, best] * lit.any(1)
    out['big_on'] = on_cov[bi, best] * lit.any(1); out['big_off'] = off_cov[bi, best] * lit.any(1)
    out['big_absd'] = np.abs(d[bi, best]) * lit.any(1)
    # detections NOT explained by the biggest cluster -> other transmitters
    inbig = same[bi, best] | (np.arange(12)[None, :] == best[:, None])
    other = lit & ~inbig
    out['n_other'] = other.sum(1)
    out['other_on_only'] = (other & np.isin(pan, ON)).sum(1)
    out['other_off'] = (other & np.isin(pan, OFF)).sum(1)
    out['other_maxscore'] = np.where(other, s, 0).max(1)
    # panels lit at all, and best-line channel/drift scatter across lit panels (strongest line only)
    lit1 = sc[..., 0] > thr
    out['n_panels_lit'] = lit1.sum(1)
    c1 = np.where(lit1, ch[..., 0], np.nan); d1 = np.where(lit1, dr[..., 0], np.nan)
    out['ch_spread'] = np.nan_to_num(np.nanmax(c1, 1) - np.nanmin(c1, 1))
    out['dr_spread'] = np.nan_to_num(np.nanmax(d1, 1) - np.nanmin(d1, 1))
    out['n_clusters_approx'] = np.round((lit / np.maximum(np.where(lit, cover * 0 + same.sum(-1), 1), 1)).sum(1), 2)
    return out


def raw_levels(u8):
    v = u8.astype(np.float32); v[v == 0] = np.nan; v[..., 64] = np.nan
    pm = np.nanmedian(v.reshape(len(v), 6, -1), 2)
    tm = np.nanmedian(v, axis=2)
    ref = np.nanmean(tm[:, OFF], 1)
    out = {'lvl_pm_on_m_off': pm[:, ON].mean(1) - pm[:, OFF].mean(1),
           'lvl_ch_on_m_off': np.mean([np.nanmedian(tm[:, p] - ref, -1) for p in ON], 0),
           'lvl_on_min_m_offmax': pm[:, ON].min(1) - pm[:, OFF].max(1),
           'lvl_panel_std': pm.std(1)}
    rel = pm - pm.mean(1, keepdims=True)
    for p in range(6):
        out[f'lvl_rel_p{p}'] = rel[:, p]
    return out


def extract6(u8):
    z, valid, _, _ = preprocess(u8)
    S, _ = drift_sums(z, valid, SLOW, PAD)
    sc, ch, dr = top_lines(S)
    out = {}
    for thr, nm in [(6.0, 'h'), (4.5, 'l')]:
        for k, v in geometry(sc, ch, dr, thr).items():
            out[f'geo{nm}_{k}'] = v
    out.update(raw_levels(u8))
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract6(u8))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat6_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
