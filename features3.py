"""Feature group 3: explicit ETI track-consistency (refinement around the best global track).

For the best global ON track (max of min-ON minus max-OFF over drifting tracks), each ON
panel is re-searched locally (+-10 channels, all slow drifts). A genuine ETI signal has
its local best line exactly on the predicted position with the same drift; the
"same drift, unconnected" trap shows systematic offsets. Also: a straight-line fit of the
three ON local positions in global time (residual), and a 2-channel-matched version.
python features3.py <train|test> [aug_seed]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from prep import preprocess
from drift import drift_sums, calibrate
from features import SLOW, PAD, D, ON, OFF

warnings.filterwarnings('ignore')
DF = np.round(np.arange(-1.5, 1.5001, 0.01), 3)
KIDX = np.abs(DF[:, None] - SLOW[None, :]).argmin(1)
DRIFTING = np.abs(DF) >= 0.045


def best_track(S, gaps):
    """returns per-cadence best (d, c0 index) of eti score and its value"""
    B, _, ND, W = S.shape
    bd = np.zeros(B); bj = np.zeros(B, int); bv = np.full(B, -1e9)
    for gap in np.unique(gaps):
        bi = np.where(gaps == gap)[0]
        L = 16 + gap
        vals = []
        for p in range(6):
            sh = np.floor(DF * p * L + 0.5).astype(int)
            j = np.arange(W)[None, :] + sh[:, None]
            ok = (j >= 0) & (j < W)
            v = S[bi, p][:, KIDX[:, None], np.clip(j, 0, W - 1)]
            vals.append(np.where(ok[None], v, 0))
        V = np.stack(vals, 1)
        eti = V[:, ON].min(1) - np.maximum(V[:, OFF].max(1), 0)
        eti = np.where(DRIFTING[None, :, None], eti, -1e9).reshape(len(bi), -1)
        i = eti.argmax(1)
        bv[bi] = eti[np.arange(len(bi)), i]
        bd[bi] = DF[i // W]; bj[bi] = i % W
    return bd, bj, bv


def extract3(u8, gaps):
    B = u8.shape[0]
    z, valid, _, _ = preprocess(u8)
    out = {}
    for tag, zz in [('', z), ('w2_', None)]:
        if tag == 'w2_':   # 2-channel matched filter (lines straddling channels / wider lines)
            zf = np.where(valid, z, 0)
            zz = np.full_like(z, np.nan)
            zz[..., :-1] = (zf[..., :-1] + zf[..., 1:]) / np.sqrt(2)
            vv = valid.copy(); vv[..., -1] = False
            S, _ = drift_sums(np.where(vv, zz, 0), vv, SLOW, PAD)
            S = calibrate(S)
        else:
            S, _ = drift_sums(z, valid, SLOW, PAD)
            S = calibrate(S)
        W = S.shape[-1]
        bd, bj, bv = best_track(S, gaps)
        out[f'{tag}t_eti'] = bv
        if tag == 'w2_':
            continue
        L = 16 + gaps
        offs, dds, loc, pred_v, gt, pos = [], [], [], [], [], []
        for p in ON:
            pc = bj + np.floor(bd * p * L + 0.5).astype(int)          # predicted index in panel p
            win = np.arange(-10, 11)
            jj = np.clip(pc[:, None] + win[None, :], 0, W - 1)          # B,21
            loc_s = S[np.arange(B)[:, None, None], p, np.arange(len(SLOW))[None, :, None], jj[:, None, :]]  # B,ND,21
            flat = loc_s.reshape(B, -1); im = flat.argmax(1)
            kd, kw = im // len(win), im % len(win)
            offs.append(win[kw]); dds.append(SLOW[kd] - bd); loc.append(flat.max(1))
            kpred = np.abs(SLOW[None, :] - bd[:, None]).argmin(1)
            pred_v.append(S[np.arange(B), p, kpred, np.clip(pc, 0, W - 1)])
            gt.append(p * L + 7.5); pos.append(pc + win[kw])
        offs, dds, loc, pred_v = map(np.stack, (offs, dds, loc, pred_v))  # 3,B
        out['t_off_absmax'] = np.abs(offs).max(0)
        out['t_off_abssum'] = np.abs(offs).sum(0)
        out['t_dd_absmax'] = np.abs(dds).max(0)
        out['t_loc_minus_pred_max'] = (loc - pred_v).max(0)
        out['t_pred_min'] = pred_v.min(0)
        out['t_loc_min'] = loc.min(0)
        # straight-line fit through local positions in global time
        gt = np.stack([np.broadcast_to(g, (B,)) for g in gt]).astype(float); pos = np.stack(pos).astype(float)
        gm, pm = gt.mean(0), pos.mean(0)
        slope = ((gt - gm) * (pos - pm)).sum(0) / ((gt - gm) ** 2).sum(0)
        resid = pos - (pm + slope * (gt - gm))
        out['t_fit_resid'] = np.abs(resid).max(0)
        out['t_fit_slope_vs_local'] = np.abs(slope - (bd + dds.mean(0)))
        out['t_abs_d'] = np.abs(bd)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    gaps = pd.read_csv(f'{D}/{split}_meta.csv').slew_gap_rows.values
    parts = []; t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract3(u8, gaps[s:s + batch]))
    F = pd.concat(parts, ignore_index=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    F.to_pickle(f'cache/feat3_{tag}.pkl'); print('done', tag, F.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], aug=int(sys.argv[2]) if len(sys.argv) > 2 else None)
