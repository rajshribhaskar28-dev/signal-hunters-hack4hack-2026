"""Precise ETI track geometry: refine per-ON-panel line positions around the best global track and
measure collinearity in global time and in-panel vs global drift consistency."""
import numpy as np
from features import FINE, GLOB, NC, ON, OFF
def refine_geometry(Mc, gaps, kk_glob, cc, win=5):
    """Mc (B,6,K,128) clipped line maps; kk_glob, cc: best global track (drift index into GLOB, channel at panel0 row0).
    returns dict of features."""
    B, _, K, _ = Mc.shape
    out = {k: np.zeros(B, np.float32) for k in ['geo_resid', 'geo_absresid', 'geo_slope_dev', 'geo_drift_spread',
                                                  'geo_inpanel_vs_global', 'geo_peak_min', 'geo_peak_sharp']}
    for i in range(B):
        g = gaps[i]; d = GLOB[kk_glob[i]]; mids = []; dps = []; pk = []; sharp = []
        for p in ON:
            c0 = cc[i] + d * p * (16 + g)
            lo = int(np.floor(c0 - win)); hi = int(np.ceil(c0 + win)) + 1
            kc = int(np.clip(round(d * 30) + 45, 0, K - 1)); klo, khi = max(0, kc - 6), min(K, kc + 7)
            sub = Mc[i, p, klo:khi, max(lo, 0):min(hi, NC)]
            if sub.size == 0:
                mids.append(c0 + d * 7.5); dps.append(d); pk.append(0); sharp.append(0); continue
            a = np.unravel_index(sub.argmax(), sub.shape)
            kd = FINE[klo + a[0]]; ch = max(lo, 0) + a[1]
            # sub-channel refinement by parabola on channel axis
            row = Mc[i, p, klo + a[0]]
            if 0 < ch < NC - 1:
                y0, y1, y2 = row[ch - 1], row[ch], row[ch + 1]; den = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            else: off = 0.0
            mids.append(ch + off + kd * 7.5); dps.append(kd); pk.append(sub.max())
            sharp.append(sub.max() - np.median(sub))
        t = np.array([p * (16 + g) + 7.5 for p in ON]); x = np.array(mids)
        out['geo_resid'][i] = x[1] - 0.5 * (x[0] + x[2])
        out['geo_absresid'][i] = abs(x[1] - 0.5 * (x[0] + x[2]))
        slope = (x[2] - x[0]) / (t[2] - t[0])
        out['geo_slope_dev'][i] = abs(slope - d)
        out['geo_drift_spread'][i] = np.ptp(dps)
        out['geo_inpanel_vs_global'][i] = abs(np.mean(dps) - slope)
        out['geo_peak_min'][i] = min(pk); out['geo_peak_sharp'][i] = min(sharp)
    return out
