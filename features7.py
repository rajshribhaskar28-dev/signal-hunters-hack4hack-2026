"""Feature group 7: line continuity (duty cycle / flicker) and ON-vs-OFF strength of the same line.

Intermittent RFI often appears at the same channel in all panels but is dashed - it switches
on and off within panels - whereas persistent RFI is a continuous line. For the two strongest
slow-drift lines per panel (from features6.top_lines) the row-by-row brightness along the
track (max over +-1 channel) is traced; duty cycle, on/off transitions, longest gap and
coefficient of variation describe continuity.
python features7.py <train|test> [aug_seed]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from prep import preprocess
from drift import drift_sums
from features import SLOW, PAD, D, ON, OFF
from features6 import top_lines, NB

warnings.filterwarnings('ignore')


def track_rows(z, valid, ch, dr):
    """z (B,6,16,128); ch, dr (B,6,NB) -> row values (B,6,NB,16) along each track, NaN if invalid."""
    B = z.shape[0]
    t = np.arange(16) - 7.5
    pos = np.floor(ch[..., None] + dr[..., None] * t + 0.5).astype(int)          # B,6,NB,16
    zf = np.where(valid, z, np.nan)
    zp = np.full((B, 6, 16, 128 + 2 * 40), np.nan, np.float32); zp[..., 40:168] = zf
    bi = np.arange(B)[:, None, None, None]; pi = np.arange(6)[None, :, None, None]; ri = np.arange(16)[None, None, None, :]
    vals = []
    for off in (-1, 0, 1):
        idx = np.clip(pos + off + 40, 0, 128 + 79)
        vals.append(zp[bi, pi, ri, idx])
    return np.nanmax(np.stack(vals), 0)


def runs(on):
    """on (...,16) bool -> transitions, longest off-run."""
    tr = (on[..., 1:] != on[..., :-1]).sum(-1)
    longest = np.zeros(on.shape[:-1]); cur = np.zeros(on.shape[:-1])
    for r in range(on.shape[-1]):
        cur = np.where(~on[..., r], cur + 1, 0); longest = np.maximum(longest, cur)
    return tr, longest


def extract7(u8):
    z, valid, _, _ = preprocess(u8)
    S, _ = drift_sums(z, valid, SLOW, PAD)
    sc, ch, dr = top_lines(S)
    R = track_rows(z, valid, ch, dr)                                 # B,6,NB,16
    on = np.nan_to_num(R, nan=-9) > 2.0
    duty = on.mean(-1)
    tr, gap = runs(on)
    mu = np.nanmean(R, -1); sd = np.nanstd(R, -1)
    cv = sd / np.maximum(np.abs(mu), 0.5)
    out = {}
    for thr, nm in [(6.0, 'h'), (4.5, 'l')]:
        lit = sc > thr                                                # B,6,NB
        def agg(a, f, fill):
            x = np.where(lit, a, np.nan).reshape(len(a), -1)
            r = f(x, 1); return np.where(np.isfinite(r), r, fill)
        out[f'cont{nm}_duty_min'] = agg(duty, np.nanmin, 1); out[f'cont{nm}_duty_mean'] = agg(duty, np.nanmean, 1)
        out[f'cont{nm}_trans_max'] = agg(tr, np.nanmax, 0); out[f'cont{nm}_trans_mean'] = agg(tr, np.nanmean, 0)
        out[f'cont{nm}_gap_max'] = agg(gap, np.nanmax, 0)
        out[f'cont{nm}_cv_max'] = agg(cv, np.nanmax, 0); out[f'cont{nm}_cv_mean'] = agg(cv, np.nanmean, 0)
        out[f'cont{nm}_n_dashed'] = (lit & (tr >= 4)).sum((1, 2))
        out[f'cont{nm}_n_continuous'] = (lit & (duty >= 0.85)).sum((1, 2))
    # same strongest line (panel-0-best channel cluster) strength ON vs OFF: use features6 geometry idea
    s1 = sc[..., 0]; c1 = ch[..., 0]
    ref = c1[np.arange(len(c1)), s1.argmax(1)]
    near = np.abs(c1 - ref[:, None]) <= 2
    st = np.where(near, s1, 0)
    out['same_line_on_mean'] = st[:, ON].mean(1); out['same_line_off_mean'] = st[:, OFF].mean(1)
    out['same_line_on_over_off'] = (st[:, ON].mean(1) + 1) / (st[:, OFF].mean(1) + 1)
    out['same_line_strength_cv'] = st.std(1) / (st.mean(1) + 1)
    out['same_line_min_over_max'] = (st.min(1) + 1) / (st.max(1) + 1)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract7(u8))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat7_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
