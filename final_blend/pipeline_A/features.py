"""Physics-informed cadence features for Signal Hunters (Hack4Hack 2026 Round 2).
All features computed from standardized excess power z (per-panel bandpass-flattened, robust-noise normalized),
with dropped rows and broadband-glitch rows masked."""
import numpy as np, pandas as pd, warnings
from scipy.ndimage import maximum_filter1d, uniform_filter
from prep import preprocess
from ops import drift_op, disp_op
warnings.filterwarnings('ignore')

NR, NC = 16, 128
ON, OFF = [0, 2, 4], [1, 3, 5]
FINE = np.arange(-45, 46) / 30.0            # narrowband drift grid, channels/row (+-1.5)
STEEP = np.concatenate([-np.arange(20, 1, -1) / 2.0, np.arange(2, 21) / 2.0])  # +-1..10 ch/row
DTS = np.array([2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 36, 44, 52, 62, 74, 88, 104])
F0 = 400

_A_fine, _n_fine, _c_fine = drift_op(FINE)
_A_steep, _n_steep, _c_steep = drift_op(STEEP)
_A_disp, _meta_disp, _c_disp = disp_op(F0, DTS)
_n_disp = np.sqrt(np.maximum(_c_disp, 1)).astype(np.float32)
_PAD = 200


def _apply(A, norm, cnt, Z, mincnt):
    S = (A @ Z) / norm[:, None]
    S[cnt < mincnt] = 0
    return S


def panel_maps(z):
    B = z.shape[0]
    Z = z.reshape(B * 6, NR * NC).T                       # (2048, B*6)
    Mf = _apply(_A_fine, _n_fine, _c_fine, Z, 8).T.reshape(B, 6, len(FINE), NC)
    Ms = _apply(_A_steep, _n_steep, _c_steep, Z, 6).T.reshape(B, 6, len(STEEP), NC)
    Sd = _apply(_A_disp, _n_disp, _c_disp, Z, 40).T.reshape(B, 6, -1)
    Zr = z[..., ::-1].reshape(B * 6, NR * NC).T            # reversed-frequency sweeps (control)
    Sr = _apply(_A_disp, _n_disp, _c_disp, Zr, 40).T.reshape(B, 6, -1)
    return Mf, Ms, Sd, Sr


def count_peaks(prof, thr, w=3):
    """count local maxima above thr along last axis (separated by >= w)"""
    mf = maximum_filter1d(prof, size=2 * w + 1, axis=-1, mode='constant', cval=-1e9)
    return ((prof >= mf) & (prof > thr)).sum(-1)


GLOB = np.arange(-180, 181) / 120.0                     # global drift grid (ch/row), decoupled from in-panel grid
_KNEAR = np.clip(np.round(GLOB * 30).astype(int) + 45, 0, len(FINE) - 1)


def eti_search(Mc, gaps):
    """Global-time ON/OFF track consistency on clipped line maps Mc (B,6,K,128).
    Track with global drift d starting at channel c (panel 0,row 0) sits at c + d*p*(16+gap) at row 0 of panel p.
    ON evidence: in-panel line SNR at nearest in-panel drift, +-1 channel. OFF veto: +-4 ch, +-1 drift step."""
    B, _, K, _ = Mc.shape
    Mon = maximum_filter1d(Mc, size=3, axis=-1)
    Moff = maximum_filter1d(maximum_filter1d(Mc, size=9, axis=-1), size=3, axis=2)
    Mex = Mc                                              # exact (no tolerance)
    names = ['minon', 'meanon', 'maxoff', 'meanoff', 'absd', 'on_mid', 'on_hi', 'off_mid', 'exact_minon', 'tight_ratio']
    out = {}
    for tag in ['a', 'nz', 'd', 'z', 'd2']:
        for nm in names: out[f'eti_{tag}_{nm}'] = np.zeros(B, np.float32)
    for nm in ['d_score', 'd2_score', 'ntracks', 'sum_score', 'z_score']: out[f'eti_{nm}'] = np.zeros(B, np.float32)
    for nm in ['npres', 'ninband', 'nabsent', 'minin', 'min2in', 'absd', 'onpres', 'offpres', 'sumv', 'npres_b', 'nabsent_b', 'n_tracks8']:
        out[f'trk_{nm}'] = np.zeros(B, np.float32)
    G = len(GLOB); cidx = np.arange(NC)
    nz = np.abs(GLOB) >= 0.075
    zr = np.abs(GLOB) < 0.04
    def pad(M):
        P = np.full((M.shape[0], 6, K, NC + 2 * _PAD), -5.0, np.float32); P[..., _PAD:_PAD + NC] = M; return P
    for g in np.unique(gaps):
        sel = np.where(gaps == g)[0]; n = len(sel); ar = np.arange(n)
        Pon, Poff, Pex = pad(Mon[sel]), pad(Moff[sel]), pad(Mex[sel])
        on = np.empty((n, 3, G, NC), np.float32); off = np.empty((n, 3, G, NC), np.float32); ex = np.empty((n, 3, G, NC), np.float32)
        for gi in range(G):
            k = _KNEAR[gi]
            for j, p in enumerate(ON):
                s = int(round(GLOB[gi] * p * (16 + g))) + _PAD
                if abs(s - _PAD) > _PAD - 1: on[:, j, gi] = -5; ex[:, j, gi] = -5; continue
                on[:, j, gi] = Pon[:, p, k, s + cidx]; ex[:, j, gi] = Pex[:, p, k, s + cidx]
            for j, p in enumerate(OFF):
                s = int(round(GLOB[gi] * p * (16 + g))) + _PAD
                if abs(s - _PAD) > _PAD - 1: off[:, j, gi] = 0; continue
                off[:, j, gi] = Poff[:, p, k, s + cidx]
        # all-panel global track occupancy (RFI persistence): tight tolerance in every panel
        allv = np.empty((n, 6, G, NC), np.float32); inb = np.zeros((6, G, NC), bool)
        for gi in range(G):
            k = _KNEAR[gi]
            for p in range(6):
                s0 = GLOB[gi] * p * (16 + g)
                mid = cidx + s0 + GLOB[gi] * 7.5
                inb[p, gi] = (mid >= 2) & (mid <= NC - 3)
                si = int(round(s0)) + _PAD
                allv[:, p, gi] = Pon[:, p, k, si + cidx] if abs(si - _PAD) <= _PAD - 1 else -5
        vc = np.minimum(allv, 15.0) * inb[None]
        tot = vc.sum(1).reshape(n, -1)
        for suffix, a in [('', tot.argmax(1))]:
            gk, cc = a // NC, a % NC
            v = allv[ar, :, gk, cc]; ib = inb[:, gk, cc].T          # (n,6)
            pres = (v > 6) & ib
            out['trk_npres'][sel] = pres.sum(1); out['trk_ninband'][sel] = ib.sum(1)
            out['trk_nabsent'][sel] = (ib & (v < 4)).sum(1)
            vin = np.where(ib, v, 99); srt = np.sort(vin, 1)
            out['trk_minin'][sel] = srt[:, 0]; out['trk_min2in'][sel] = srt[:, 1]
            out['trk_absd'][sel] = np.abs(GLOB[gk]); out['trk_sumv'][sel] = tot.max(1)
            out['trk_onpres'][sel] = pres[:, ON].sum(1); out['trk_offpres'][sel] = pres[:, OFF].sum(1)
        # track with most panels present (>8), any drift
        cnt = (((allv > 8) & inb[None]).sum(1)).reshape(n, -1)
        a = cnt.argmax(1); gk, cc = a // NC, a % NC
        v = allv[ar, :, gk, cc]; ib = inb[:, gk, cc].T
        out['trk_npres_b'][sel] = cnt.max(1); out['trk_nabsent_b'][sel] = (ib & (v < 4)).sum(1)
        tmax = np.where(cnt >= 2, tot, 0).reshape(n, G, NC).max(1)
        out['trk_n_tracks8'][sel] = count_peaks(tmax, 40.0, w=4)
        minon = on.min(1); maxoff = off.max(1); exmin = ex.min(1)
        diff = minon - np.maximum(maxoff, 0)
        def fill(tag, score, kmask):
            sc = np.where(kmask[None, :, None], score, -1e9).reshape(n, -1)
            a = sc.argmax(1); kk, cc = a // NC, a % NC
            o = np.sort(on[ar, :, kk, cc], 1); f_ = np.sort(off[ar, :, kk, cc], 1)
            out[f'eti_{tag}_minon'][sel] = o[:, 0]; out[f'eti_{tag}_meanon'][sel] = o.mean(1)
            out[f'eti_{tag}_maxoff'][sel] = f_[:, 2]; out[f'eti_{tag}_meanoff'][sel] = f_.mean(1)
            out[f'eti_{tag}_absd'][sel] = np.abs(GLOB[kk])
            out[f'eti_{tag}_on_mid'][sel] = o[:, 1]; out[f'eti_{tag}_on_hi'][sel] = o[:, 2]
            out[f'eti_{tag}_off_mid'][sel] = f_[:, 1]
            e = exmin[ar, kk, cc]; out[f'eti_{tag}_exact_minon'][sel] = e
            if tag in ('d', 'nz'):
                out.setdefault(f'_kk_{tag}', np.zeros(B, int))[sel] = kk; out.setdefault(f'_cc_{tag}', np.zeros(B, int))[sel] = cc
            out[f'eti_{tag}_tight_ratio'][sel] = e / (np.abs(o[:, 0]) + 1)
            return sc.max(1)
        fill('a', minon, np.ones(G, bool))
        fill('nz', minon, nz)
        out['eti_d_score'][sel] = fill('d', diff, nz)
        out['eti_z_score'][sel] = fill('z', diff, zr)
        sc2 = on.mean(1) - np.maximum(maxoff, 0) - 0.5 * (on.max(1) - minon)
        out['eti_d2_score'][sel] = fill('d2', sc2, nz)
        out['eti_sum_score'][sel] = np.where(nz[None, :, None], on.sum(1) / np.sqrt(3) - off.max(1), -1e9).reshape(n, -1).max(1)
        dm = np.where(nz[None, :, None], diff, -1e9).max(1)
        out['eti_ntracks'][sel] = count_peaks(dm, 4.0, w=3)
    return out

def level_features(v, bad):
    """ON/OFF relative intensity & patchiness. v uint8 (B,6,16,128); bad (B,6,16) masked rows."""
    B = v.shape[0]
    L = v.astype(np.float32) / 24.0
    L[..., 64] = 0.5 * (L[..., 63] + L[..., 65])
    L[bad] = np.nan
    f = {}
    lvl = np.nanmedian(L, axis=(2, 3))                       # (B,6) panel level (log2)
    lvl = np.nan_to_num(lvl - np.nanmedian(lvl, axis=1, keepdims=True))
    for p in range(6): f[f'lvl_p{p}'] = lvl[:, p]
    f['lvl_onoff'] = lvl[:, ON].mean(1) - lvl[:, OFF].mean(1)
    f['lvl_onmin_offmax'] = lvl[:, ON].min(1) - lvl[:, OFF].max(1)
    f['lvl_spread'] = lvl.max(1) - lvl.min(1)
    # alternating-pattern strength relative to non-alternating gain wander
    alt = np.array([1, -1, 1, -1, 1, -1], np.float32) / 6
    f['lvl_alt'] = (lvl * alt).sum(1)
    f['lvl_resid'] = np.std(lvl - np.outer(f['lvl_alt'], alt * 6), axis=1)
    f['lvl_alt_t'] = f['lvl_alt'] / (f['lvl_resid'] + 0.01)
    # per-channel spectra (median over time) relative to OFF reference, gain removed
    spec = np.nanmedian(L, axis=2)                            # (B,6,128)
    spec = np.where(np.isnan(spec), np.nanmedian(spec, axis=-1, keepdims=True), spec)
    out_std = np.zeros((B, 6), np.float32); out_max = np.zeros((B, 6), np.float32); out_frac = np.zeros((B, 6), np.float32)
    noise = np.zeros((B, 6), np.float32)
    for p in range(6):
        others = [q for q in OFF if q != p]
        ref = spec[:, others].mean(1)
        r = spec[:, p] - ref
        r = r - np.median(r, axis=-1, keepdims=True)
        hf = np.median(np.abs(np.diff(r, axis=-1)), axis=-1) * 1.4826 / np.sqrt(2) + 1e-4
        rs = uniform_filter(r, size=(1, 6), mode='nearest')
        rs = rs - np.median(rs, -1, keepdims=True)
        out_std[:, p] = rs.std(-1) / hf; out_max[:, p] = rs.max(-1) / hf
        out_frac[:, p] = (rs > 2 * hf[:, None] / np.sqrt(6)).mean(-1); noise[:, p] = hf
    # time variation: row means relative to panel (gain-free within panel)
    rowm = np.nanmean(np.exp2(L), axis=-1)                    # (B,6,16)
    rowm = rowm / np.nanmedian(rowm, axis=-1, keepdims=True)
    rstd = np.nan_to_num(np.nanstd(rowm, axis=-1))
    for nm, a in [('spec_std', out_std), ('spec_max', out_max), ('spec_frac', out_frac), ('row_var', rstd), ('specnoise', noise)]:
        for p in range(6): f[f'{nm}_p{p}'] = a[:, p]
        f[f'{nm}_onmean'] = a[:, ON].mean(1); f[f'{nm}_offmean'] = a[:, OFF].mean(1)
        f[f'{nm}_diff'] = a[:, ON].mean(1) - a[:, OFF].mean(1); f[f'{nm}_onmin'] = a[:, ON].min(1)
    # linear power excess ON vs OFF, per channel block: patchy broadband excess
    P = np.exp2(L)
    blk = np.nanmean(P.reshape(B, 6, 16, 16, 8), axis=(2, 4))  # (B,6,16 blocks)
    blk = blk / np.nanmedian(blk, axis=(1, 2), keepdims=True)
    offref = np.nanmean(blk[:, OFF], axis=1)
    on_ex = blk[:, ON] / offref[:, None] - 1
    on_ex = np.nan_to_num(on_ex)
    f['blk_ex_mean'] = on_ex.mean((1, 2)); f['blk_ex_max'] = on_ex.max((1, 2))
    f['blk_ex_std'] = on_ex.std(2).mean(1); f['blk_ex_minpanel'] = on_ex.mean(2).min(1)
    return f


def cadence_features(v, gaps):
    """v: (B,6,16,128) uint8, gaps: (B,) slew_gap_rows -> DataFrame of features"""
    z, valid, glitch, drop, dev = preprocess(v)
    B = z.shape[0]
    f = {}
    pf = {}  # per-panel features (B,6)
    zc = np.clip(z, -6, 30)
    pf['std'] = zc.std(axis=(2, 3))
    pf['f3'] = (z > 3).mean(axis=(2, 3)); pf['f5'] = (z > 5).mean(axis=(2, 3))
    pf['f8'] = (z > 8).mean(axis=(2, 3))
    flat = z.reshape(B, 6, -1)
    pf['p99'] = np.quantile(flat, 0.99, axis=-1)
    pf['max'] = flat.max(-1)
    pf['top32'] = -np.sort(-flat, axis=-1)[..., :32].mean(-1)
    pf['nglitch'] = glitch.sum(-1).astype(np.float32); pf['ndrop'] = drop.sum(-1).astype(np.float32)
    pf['devmax'] = np.nan_to_num(np.nanmax(np.abs(dev), -1))
    nval = valid.sum(-1).clip(1)
    rowmean = zc.mean(-1)                                   # broadband rows (B,6,16)
    pf['rowmax'] = rowmean.max(-1) * np.sqrt(NC)
    colmean = zc.sum(2) / np.sqrt(nval)[..., None]          # zero-drift column SNR (B,6,128)
    pf['colmax'] = colmean.max(-1)
    pf['ncol6'] = count_peaks(colmean, 6.0)
    # coarse texture (scintillation / blotches)
    sm = uniform_filter(zc, size=(1, 1, 4, 8), mode='nearest')
    pf['sm_std'] = sm.std(axis=(2, 3)); pf['sm_f1'] = (sm > 1.0).mean(axis=(2, 3))
    pf['sm_max'] = sm.max(axis=(2, 3))
    sm2 = uniform_filter(zc, size=(1, 1, 8, 32), mode='nearest')
    pf['sm2_std'] = sm2.std(axis=(2, 3))
    prof = zc.mean(2)                                       # (B,6,128) time-avg spectrum
    pf['prof_std'] = uniform_filter(prof, size=(1, 1, 8)).std(-1)
    pf['row_std'] = rowmean.std(-1)
    # 2D FFT periodicity (radar lattice)
    F = np.abs(np.fft.rfft2(np.clip(z, 0, 8)))
    F[..., 0, 0] = 0
    Fm = F.reshape(B, 6, -1)
    pf['fft_peak'] = Fm.max(-1) / (Fm.mean(-1) + 1e-6)
    F2 = F.copy(); F2[..., :, 0] = 0; F2[..., 0, :] = 0   # exclude pure row/col structure
    F2m = F2.reshape(B, 6, -1); pf['fft_peak2'] = F2m.max(-1) / (F2m.mean(-1) + 1e-6)
    # line search maps
    Mf, Ms, Sd, Sr = panel_maps(z)
    Kf = len(FINE)
    mfk = Mf.max(-1)                                        # (B,6,K)
    pf['nb_max'] = mfk.max(-1)
    pf['nb_absd'] = np.abs(FINE[mfk.argmax(-1)])
    pf['nb_d'] = FINE[mfk.argmax(-1)]
    zmask = np.abs(FINE) < 1.5 / 15.0
    pf['nb_zero'] = mfk[..., zmask].max(-1)
    pf['nb_nz'] = mfk[..., ~zmask].max(-1)
    nbprof = Mf.max(2)                                      # (B,6,128)
    pf['nb_n6'] = count_peaks(nbprof, 6.0); pf['nb_n10'] = count_peaks(nbprof, 10.0)
    pf['nb_ch'] = nbprof.argmax(-1).astype(np.float32)
    msk = Ms.max(-1)
    pf['st_max'] = msk.max(-1); pf['st_absd'] = np.abs(STEEP[msk.argmax(-1)])
    stprof = Ms.max(2)
    pf['st_n6'] = count_peaks(stprof, 6.0, w=4); pf['st_n10'] = count_peaks(stprof, 10.0, w=4)
    # number of distinct steep tracks at the best steep drift (radar: parallel repeated sweeps)
    bk = msk.argmax(-1)
    best_line = np.take_along_axis(Ms, bk[..., None, None], axis=2)[:, :, 0]
    pf['st_rep'] = count_peaks(best_line, 6.0, w=4)
    pf['st_pos'] = Ms[:, :, len(STEEP) // 2:].max(axis=(2, 3)); pf['st_neg'] = Ms[:, :, :len(STEEP) // 2].max(axis=(2, 3))
    # dispersion
    pf['dm_max'] = Sd.max(-1); pf['dm_rev'] = Sr.max(-1)
    am = Sd.argmax(-1); pf['dm_dt'] = _meta_disp[am, 0].astype(np.float32)
    # repeats at best Dt: count separated peaks in t0 series
    reps = np.zeros((B, 6), np.float32)
    for j, Dt in enumerate(DTS):
        idx = np.where(_meta_disp[:, 0] == Dt)[0]
        ser = Sd[..., idx]
        c = count_peaks(ser, 5.0, w=2)
        m = pf['dm_dt'] == Dt
        reps[m] = c[m]
    pf['dm_rep'] = reps
    pf['dm_n5'] = (Sd > 6).sum(-1).astype(np.float32)
    # collect per-panel and ON/OFF aggregates
    for k, a in pf.items():
        a = a.astype(np.float32)
        for p in range(6):
            f[f'{k}_p{p}'] = a[:, p]
        on, off = a[:, ON], a[:, OFF]
        f[f'{k}_onmean'] = on.mean(1); f[f'{k}_onmin'] = on.min(1); f[f'{k}_onmax'] = on.max(1)
        f[f'{k}_offmean'] = off.mean(1); f[f'{k}_offmax'] = off.max(1); f[f'{k}_offmin'] = off.min(1)
        f[f'{k}_diff'] = on.mean(1) - off.mean(1)
        f[f'{k}_allmax'] = a.max(1)
        srt = np.sort(a, 1)
        f[f'{k}_2nd'] = srt[:, -2]
    # panel-count style features
    for k, thr in [('nb_max', 8), ('st_max', 8), ('dm_max', 8), ('colmax', 8), ('nb_nz', 8), ('sm_std', 0.6)]:
        a = pf[k] > thr
        f[f'{k}_cnt_on'] = a[:, ON].sum(1); f[f'{k}_cnt_off'] = a[:, OFF].sum(1)
        # longest consecutive run of panels
        run = np.zeros(B); cur = np.zeros(B)
        for p in range(6):
            cur = np.where(a[:, p], cur + 1, 0); run = np.maximum(run, cur)
        f[f'{k}_run'] = run
    # persistence of narrowband lines across panels at same channel (zero-ish drift)
    pr = maximum_filter1d(nbprof, size=5, axis=-1)
    allp = pr.min(1); f['nb_persist_all'] = allp.max(-1)
    f['nb_persist_5of6'] = np.sort(pr, axis=1)[:, 1].max(-1)
    f['nb_persist_on'] = pr[:, ON].min(1).max(-1); f['nb_persist_off'] = pr[:, OFF].min(1).max(-1)
    # global ETI track search
    zc6 = np.clip(z, -4, 6).reshape(B * 6, NR * NC).T
    Mc = _apply(_A_fine, _n_fine, _c_fine, zc6, 8).T.reshape(B, 6, len(FINE), NC)
    es = eti_search(Mc, gaps)
    from geom import refine_geometry
    for tag in ('d', 'nz'):
        gfe = refine_geometry(Mc, gaps, es.pop(f'_kk_{tag}'), es.pop(f'_cc_{tag}'))
        f.update({k.replace('geo_', f'geo_{tag}_'): v for k, v in gfe.items()})
    f.update(es)
    f.update(level_features(v, drop | glitch))
    return pd.DataFrame(f)


def extract(X, meta, batch=200, n_jobs=2):
    from joblib import Parallel, delayed
    gaps = meta['slew_gap_rows'].values
    def run(s):
        return cadence_features(np.asarray(X[s:s + batch]), gaps[s:s + batch])
    parts = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in range(0, len(X), batch))
    F = pd.concat(parts, ignore_index=True)
    F['slew_gap_rows'] = gaps
    return F
