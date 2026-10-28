"""Physics-informed cadence features for Signal Hunters.

Usage: python features.py <train|test> [aug_seed]
Writes cache/feat_<split>.parquet
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter, maximum_filter1d
from prep import preprocess
from drift import drift_sums, calibrate

warnings.filterwarnings('ignore')
CALIB = os.environ.get('SH_CALIB', '1') == '1'   # 0 -> raw (uncalibrated) detector scores
D = os.environ.get('SH_DATA', 'data')
ON, OFF = [0, 2, 4], [1, 3, 5]

# ---------------- drift grids ----------------
SLOW = np.round(np.arange(-1.5, 1.5001, 0.0625), 4)          # ETI / RFI regime (ch per row)
FAST = np.concatenate([np.arange(-4, -1.5, 0.25), np.arange(1.75, 4.001, 0.25)])
STEEP = np.concatenate([np.arange(-16, -4, 0.5), np.arange(4.5, 16.001, 0.5)])
PAD = 8

# ---------------- dispersion family ----------------
FLO = 0.85
_f = FLO + (1 - FLO) * np.arange(128) / 127.0
G = (_f ** -2 - 1) / (FLO ** -2 - 1)                         # 1 at ch0 (low freq, latest), 0 at ch127
AMPS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40])
NSUB = 32                                                    # 4-channel sub-bands


def panel_aggs(name, arr, out):
    """arr (B,6) -> per-panel + ON/OFF aggregates"""
    for p in range(6):
        out[f'{name}_p{p}'] = arr[:, p]
    on, off = np.sort(arr[:, ON], 1), np.sort(arr[:, OFF], 1)
    out[f'{name}_onmin'], out[f'{name}_onmed'], out[f'{name}_onmax'] = on[:, 0], on[:, 1], on[:, 2]
    out[f'{name}_offmin'], out[f'{name}_offmed'], out[f'{name}_offmax'] = off[:, 0], off[:, 1], off[:, 2]
    out[f'{name}_onmin_m_offmax'] = on[:, 0] - off[:, 2]
    out[f'{name}_onmax_m_offmax'] = on[:, 2] - off[:, 2]
    srt = np.sort(arr, 1)
    out[f'{name}_all2nd'] = srt[:, 1]       # 5-of-6 level (allows one faded panel)
    out[f'{name}_allmin'] = srt[:, 0]


def count_peaks(prof, thr, win=5):
    """prof (..., C): number of local maxima above thr separated by win."""
    mx = maximum_filter1d(prof, size=win, axis=-1, mode='constant', cval=-1e9)
    return ((prof >= mx) & (prof > thr)).sum(-1)


def dedisperse(z, valid):
    """Return D[b,p,a,t0] dedispersed significance on 32 sub-bands."""
    B = z.shape[0]
    zz = np.where(valid, z, 0).reshape(B, 6, 16, NSUB, 4).sum(-1)
    nn = valid.reshape(B, 6, 16, NSUB, 4).sum(-1).astype(np.float32)
    g = G.reshape(NSUB, 4).mean(1)
    TP = 45
    zp = np.zeros((B, 6, 16 + 2 * TP, NSUB), np.float32); zp[:, :, TP:TP + 16] = zz
    npd = np.zeros_like(zp); npd[:, :, TP:TP + 16] = nn
    t0s = np.arange(-41, 16)             # arrival time at the highest frequency
    out = np.zeros((B, 6, len(AMPS), len(t0s)), np.float32)
    cidx = np.arange(NSUB)[None, :]
    for ai, A in enumerate(AMPS):
        delay = np.floor(A * g + 0.5).astype(int)
        tidx = t0s[:, None] + delay[None, :] + TP          # (T0, NSUB)
        s = zp[:, :, tidx, cidx].sum(-1)
        n = npd[:, :, tidx, cidx].sum(-1)
        out[:, :, ai] = np.where(n >= 64, s / np.sqrt(np.maximum(n, 1)), 0)
    return out, t0s


def global_tracks(S, gaps, out):
    """Combine per-panel slow drift sums S (B,6,ND,W) along one straight track in global time.
    Global time of panel p mid = p*(16+gap)+7.5, so track channel in panel p = c0 + d*p*L."""
    B, _, ND, W = S.shape
    dfine = np.round(np.arange(-1.5, 1.5001, 0.01), 3)
    kidx = np.abs(dfine[:, None] - SLOW[None, :]).argmin(1)
    drifting = np.abs(dfine) >= 0.045
    res = {k: np.zeros(B, np.float32) for k in [
        'g_minon', 'g_minon_offmax', 'g_minon_d', 'g_minon_onsum',
        'g_eti', 'g_eti_minon', 'g_eti_offmax', 'g_eti_d', 'g_eti_spread',
        'g_onsum', 'g_onsum_offmax', 'g_onsum_minon',
        'g0_minon', 'g0_minon_offmax', 'g0_eti',
        'g_min6', 'g_2nd6', 'g_min6_d', 'g_eti_on_rank']}
    for gap in np.unique(gaps):
        bi = np.where(gaps == gap)[0]
        if len(bi) == 0:
            continue
        L = 16 + gap
        Sg = S[bi]                                          # b,6,ND,W
        vals = []
        for p in range(6):
            sh = np.floor(dfine * p * L + 0.5).astype(int)  # NDf
            j = np.arange(W)[None, :] + sh[:, None]          # NDf,W
            ok = (j >= 0) & (j < W)
            jc = np.clip(j, 0, W - 1)
            v = Sg[:, p][:, kidx[:, None], jc]               # b,NDf,W
            vals.append(np.where(ok[None], v, 0))
        V = np.stack(vals, 1)                                # b,6,NDf,W
        on = V[:, ON]; off = V[:, OFF]
        minon = on.min(1); offmax = off.max(1); onsum = on.sum(1) / np.sqrt(3)
        eti = minon - np.maximum(offmax, 0)
        all6 = np.sort(V, 1)
        min6 = all6[:, 0]; sec6 = all6[:, 1]
        flat = lambda a: a.reshape(len(bi), -1)
        def take(a, idx):
            return flat(a)[np.arange(len(bi)), idx]
        dgrid = np.repeat(dfine, W)
        for mask_d, pref in [(drifting, 'g'), (~drifting, 'g0')]:
            m = np.repeat(mask_d, W)[None, :]
            mo = np.where(m, flat(minon), -1e9); im = mo.argmax(1)
            res[f'{pref}_minon'][bi] = take(minon, im)
            res[f'{pref}_minon_offmax'][bi] = take(offmax, im)
            et = np.where(m, flat(eti), -1e9); ie = et.argmax(1)
            res[f'{pref}_eti'][bi] = take(eti, ie)
            if pref == 'g':
                res['g_minon_d'][bi] = np.abs(dgrid[im])
                res['g_minon_onsum'][bi] = take(onsum, im)
                res['g_eti_minon'][bi] = take(minon, ie)
                res['g_eti_offmax'][bi] = take(offmax, ie)
                res['g_eti_d'][bi] = np.abs(dgrid[ie])
                onv = np.stack([take(on[:, q], ie) for q in range(3)], 1)
                res['g_eti_spread'][bi] = onv.max(1) - onv.min(1)
                os_ = np.where(m, flat(onsum), -1e9); io = os_.argmax(1)
                res['g_onsum'][bi] = take(onsum, io)
                res['g_onsum_offmax'][bi] = take(offmax, io)
                res['g_onsum_minon'][bi] = take(minon, io)
        i6 = flat(min6).argmax(1)
        res['g_min6'][bi] = take(min6, i6)
        res['g_min6_d'][bi] = np.abs(dgrid[i6])
        res['g_2nd6'][bi] = flat(sec6).max(1)
    out.update(res)


def extract(u8, gaps):
    B = u8.shape[0]
    z, valid, glitch, sig = preprocess(u8)
    out = {}
    zf = np.where(valid, z, 0)
    # ---- basic pixel statistics (robust, panel-normalised) ----
    panel_aggs('frac3', (zf > 3).mean((2, 3)), out)
    panel_aggs('frac5', (zf > 5).mean((2, 3)), out)
    panel_aggs('zmax', zf.max((2, 3)), out)
    panel_aggs('pos_mean', np.clip(zf, 0, None).mean((2, 3)), out)
    # ---- smoothed texture (scintillation patches) ----
    for (ft, ff), nm in [((3, 8), 'sm38'), ((5, 16), 'sm516'), ((2, 4), 'sm24')]:
        sm = uniform_filter(zf, size=(1, 1, ft, ff), mode='constant')
        cnt = uniform_filter(valid.astype(np.float32), size=(1, 1, ft, ff), mode='constant')
        sm = np.where(cnt > 0.3, sm / np.maximum(cnt, 1e-3), 0) * np.sqrt(ft * ff)
        panel_aggs(f'{nm}_max', sm.max((2, 3)), out)
        panel_aggs(f'{nm}_std', sm.std((2, 3)), out)
        panel_aggs(f'{nm}_frac2', (sm > 2.5).mean((2, 3)), out)
    # channel occupancy of temporally averaged excess
    spec = zf.sum(2) / np.sqrt(np.maximum(valid.sum(2), 1))         # B,6,128
    panel_aggs('spec_max', spec.max(2), out)
    panel_aggs('spec_frac3', (spec > 3).mean(2), out)
    # ---- drift search ----
    S, N = drift_sums(z, valid, SLOW, PAD)                           # B,6,NS,W
    if CALIB: S = calibrate(S)
    absS = np.abs(SLOW)
    zero = absS <= 0.0625 + 1e-6
    slow = ~zero
    best_all_slow = S.max(-1)                                        # B,6,NS
    panel_aggs('line_zero', best_all_slow[:, :, zero].max(-1), out)
    panel_aggs('line_slow', best_all_slow[:, :, slow].max(-1), out)
    slow_best = best_all_slow.max(-1)
    sd = SLOW[best_all_slow.argmax(-1)]
    panel_aggs('line_slow_absd', np.abs(sd), out)
    # signed drift of best slow line in ON panels: consistency
    out['slow_d_on_std'] = np.std(sd[:, ON], 1)
    out['slow_d_on_signagree'] = np.abs(np.sign(sd[:, ON]).sum(1))
    prof_slow = S.max(2)                                             # B,6,W (best over drift per channel)
    panel_aggs('nlines6', count_peaks(prof_slow, 6.0), out)
    panel_aggs('nlines4', count_peaks(prof_slow, 4.5), out)
    # stationary persistence at the same channel across panels
    prof_zero = maximum_filter1d(S[:, :, zero].max(2), 3, axis=-1)  # B,6,W
    srt = np.sort(prof_zero, 1)
    out['stat_min6'] = srt[:, 0].max(-1); out['stat_2nd6'] = srt[:, 1].max(-1); out['stat_3rd6'] = srt[:, 2].max(-1)
    onz, offz = prof_zero[:, ON], prof_zero[:, OFF]
    out['stat_onmin_m_offmax'] = (onz.min(1) - offz.max(1)).max(-1)
    out['stat_onmin'] = onz.min(1).max(-1)
    # n panels with a stationary line at the channel of the strongest stationary line
    jbest = prof_zero.max(1).argmax(-1)
    vb = prof_zero[np.arange(B), :, jbest]                           # B,6
    out['stat_npanels6'] = (vb > 6).sum(1); out['stat_npanels4'] = (vb > 4.5).sum(1)
    out['stat_on_n'] = (vb[:, ON] > 4.5).sum(1); out['stat_off_n'] = (vb[:, OFF] > 4.5).sum(1)
    # same for slow drift lines (best drift per channel)
    jb2 = prof_slow.max(1).argmax(-1)
    vb2 = prof_slow[np.arange(B), :, jb2]
    out['slowch_npanels4'] = (vb2 > 4.5).sum(1)
    out['slowch_on_n'] = (vb2[:, ON] > 4.5).sum(1); out['slowch_off_n'] = (vb2[:, OFF] > 4.5).sum(1)
    # ---- global ON/OFF track consistency (ETI test) ----
    global_tracks(S, gaps, out)
    out['g_eti_vs_indep'] = out['g_minon'] - np.sort(best_all_slow[:, :, slow].max(-1)[:, ON], 1)[:, 0]
    del S, N
    # ---- fast & steep lines (satellite, radar) ----
    Sf, _ = drift_sums(z, valid, FAST, 70)
    Ss, _ = drift_sums(z, valid, STEEP, 260)
    if CALIB: Sf = calibrate(Sf); Ss = calibrate(Ss)
    fbest = Sf.max(-1); sbest = Ss.max(-1)
    panel_aggs('line_fast', fbest.max(-1), out)
    panel_aggs('line_steep', sbest.max(-1), out)
    panel_aggs('line_fast_absd', np.abs(FAST[fbest.argmax(-1)]), out)
    panel_aggs('line_steep_absd', np.abs(STEEP[sbest.argmax(-1)]), out)
    steep_prof = Ss.max(2)
    panel_aggs('nsteep5', count_peaks(steep_prof, 5.0, win=9), out)
    panel_aggs('nsteep4', count_peaks(steep_prof, 4.2, win=9), out)
    fast_prof = Sf.max(2)
    panel_aggs('nfast5', count_peaks(fast_prof, 5.0, win=7), out)
    # sign of steep drift: positive vs negative tracks count (sawtooth)
    pos = Ss[:, :, STEEP > 0].max((2, 3)); neg = Ss[:, :, STEEP < 0].max((2, 3))
    panel_aggs('steep_posneg_min', np.minimum(pos, neg), out)
    steep_any = np.maximum(sbest.max(-1), fbest.max(-1))
    out['n_panels_steep6'] = (steep_any > 6).sum(1)
    out['n_panels_steep5'] = (steep_any > 5).sum(1)
    # consecutive-panel confinement for satellites
    hi = steep_any > 6
    out['steep_consec_only'] = ((hi.sum(1) >= 1) & (hi.sum(1) <= 2)).astype(np.float32)
    del Sf, Ss
    # ---- dedispersion (pulsar / frb) ----
    Dd, t0s = dedisperse(z, valid)                                   # B,6,A,T
    if CALIB: Dd = calibrate(Dd)
    disp = Dd[:, :, 2:]                                              # A>=2
    dbest = disp.max((2, 3))
    panel_aggs('disp_best', dbest, out)
    panel_aggs('disp_A', AMPS[2:][disp.max(-1).argmax(-1)], out)
    panel_aggs('bb_undisp', Dd[:, :, :2].max((2, 3)), out)
    ia = disp.max(-1).argmax(-1)                                     # B,6
    tser = np.take_along_axis(disp, ia[:, :, None, None], 2)[:, :, 0]   # B,6,T
    panel_aggs('disp_npulse5', count_peaks(tser, 5.0, win=3), out)
    panel_aggs('disp_npulse4', count_peaks(tser, 4.0, win=3), out)
    panel_aggs('disp_n_t_above3', (tser > 3).sum(-1), out)
    out['disp_on_count5'] = (dbest[:, ON] > 5).sum(1)
    out['disp_off_count5'] = (dbest[:, OFF] > 5).sum(1)
    # disp vs best steep straight line: curvature evidence
    out['disp_minus_steep_onmax'] = dbest[:, ON].max(1) - np.maximum(out['line_steep_onmax'], out['line_fast_onmax'])
    out['slew_gap_rows'] = gaps.astype(np.float32)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def run(split, batch=200, aug=None):
    X = np.load(f'{D}/{split}_cadences.npy', mmap_mode='r')
    meta = pd.read_csv(f'{D}/{split}_meta.csv')
    gaps = meta.slew_gap_rows.values
    parts = []
    t0 = time.time()
    for s in range(0, len(X), batch):
        u8 = np.asarray(X[s:s + batch])
        if aug is not None:
            from augment import pks_like
            u8 = pks_like(u8, np.random.default_rng(aug * 100003 + s), seed=aug)
        parts.append(extract(u8, gaps[s:s + batch]))
        if s % 2000 == 0:
            print(split, s, f'{time.time() - t0:.0f}s', flush=True)
    F = pd.concat(parts, ignore_index=True)
    os.makedirs('cache', exist_ok=True)
    tag = split if aug is None else f'{split}_aug{aug}'
    pre = 'feat' if CALIB else 'featraw'
    F.to_pickle(f'cache/{pre}_{tag}.pkl')
    print('done', tag, F.shape, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    split = sys.argv[1]
    aug = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(split, aug=aug)
