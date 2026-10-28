"""Global-time CNN for Signal Hunters, GPU version (Colab / Kaggle).

Usage (Colab, GPU runtime):
    python gpu_global_cnn.py --zip /content/signal-hunters-round2_1.zip

Output: gpu_cnn_out.npz with
    hold_c : probabilities on fold-0 validation rows (clean)        [3600 x 9]
    hold_a : probabilities on the same rows, PKS-like augmented      [3600 x 9]
    test   : probabilities on the 10,000 test cadences (seed mean)   [10000 x 9]
Fold 0 = StratifiedKFold(5, shuffle=True, random_state=42), identical to the CPU pipeline,
so the output can be blended and validated against the existing models.
"""
import argparse, os, time, zipfile, glob
import numpy as np, pandas as pd
from scipy.ndimage import median_filter

CLASSES = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass', 'radar_chirp',
           'rfi_intermittent', 'rfi_persistent', 'noise']
OFF = [1, 3, 5]


# ---------------- preprocessing (identical to the CPU pipeline: prep.py / augment.py / cnn_inputs.py) -------------
def preprocess(v):
    v = v.astype(np.float32)
    drop = (v == 0).all(-1)
    v[..., 64] = 0.5 * (v[..., 63] + v[..., 65])
    v[drop] = np.nan
    L = v / 24.0
    rowmed = np.nanmedian(L, axis=-1)
    pmed = np.nanmedian(rowmed, axis=-1, keepdims=True)
    glitch = np.abs(rowmed - pmed) > 0.35
    bad = drop | glitch
    L[bad] = np.nan
    P = np.exp2(L - 4)
    bp = np.nanmedian(P, axis=2)
    bp = np.where(np.isnan(bp), np.nanmedian(bp, axis=-1, keepdims=True), bp)
    bp = median_filter(bp, size=(1, 1, 9), mode='nearest')
    N = P / bp[:, :, None, :] - 1
    med = np.nanmedian(N, axis=(2, 3), keepdims=True)
    mad = np.nanmedian(np.abs(N - med), axis=(2, 3), keepdims=True) * 1.4826 + 1e-6
    z = np.nan_to_num((N - med) / mad, nan=0.0).astype(np.float32)
    return z, ~bad


def bandpass_profile(X, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), min(n, len(X)), replace=False))
    a = np.asarray(X[idx]).astype(np.float32); a[a == 0] = np.nan
    a[..., 64] = 0.5 * (a[..., 63] + a[..., 65])
    prof = np.nanmedian(a, axis=(0, 1, 2))
    return prof - np.nanmedian(prof)


def pksify(v, rng, bp_delta, gain_sd=1.9, noise_k=0.5, noise_sd=11.5, p_drop=0.0025, p_glitch=0.009, tilt_sd=3.0):
    B = v.shape[0]
    x = v.astype(np.float32); drop0 = (v == 0)
    x[..., 64] = 0.5 * (x[..., 63] + x[..., 65])
    scale = rng.uniform(0.7, 1.3, size=(B, 1)); bp = scale * bp_delta[None, :]
    tilt = rng.normal(0, tilt_sd, size=(B, 1)) * np.linspace(-1, 1, 128)[None]
    x += (bp + tilt)[:, None, None, :]
    x += rng.normal(0, gain_sd, size=(B, 6, 1, 1))
    n = 24.0 * np.log2(rng.gamma(noise_k, 1.0, size=x.shape).astype(np.float32) + 1e-30)
    n = np.clip(n, -400, None)
    x += (n - n.mean()) / (n.std() + 1e-6) * noise_sd
    gl = rng.random((B, 6, 16)) < p_glitch
    amp = rng.uniform(10, 50, size=gl.sum()); sign = np.where(rng.random(gl.sum()) < 0.15, -0.4, 1.0)
    x[gl] += (amp * sign)[:, None]
    x = np.clip(np.round(x), 1, 255)
    dr = rng.random((B, 6, 16)) < p_drop
    x[dr] = 0; x[drop0] = 0
    return x.astype(np.uint8)


def cnn_inputs(v):
    z, valid = preprocess(v)
    a = np.clip(z, -3, 12) / 3.0
    L = v.astype(np.float32) / 24.0; L[..., 64] = 0.5 * (L[..., 63] + L[..., 65]); L[~valid] = np.nan
    spec = np.nanmedian(L, axis=2)
    ref = np.nanmean(spec[:, OFF], axis=1)
    r = L - ref[:, None, None, :]
    r = r - np.nanmedian(r[:, OFF], axis=(1, 2, 3))[:, None, None, None]
    r = np.clip(np.nan_to_num(r), -1, 1) * 3
    return np.concatenate([a, r], axis=1).astype(np.float16)          # (B,12,16,128)


def build(X, aug_seed=None, delta=None, batch=200):
    out = np.zeros((len(X), 12, 16, 128), np.float16)
    for s in range(0, len(X), batch):
        v = np.asarray(X[s:s + batch])
        if aug_seed is not None: v = pksify(v, np.random.default_rng(aug_seed * 100000 + s), delta)
        out[s:s + len(v)] = cnn_inputs(v)
    return out


# ---------------- model ----------------
def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--out', default='gpu_cnn_out.npz')
    a = ap.parse_args()
    D = a.data
    if D is None:
        os.makedirs('/content/sh_data' if os.path.isdir('/content') else 'sh_data', exist_ok=True)
        root = '/content/sh_data' if os.path.isdir('/content') else 'sh_data'
        if not glob.glob(f'{root}/**/train_meta.csv', recursive=True):
            print('unzipping...', flush=True); zipfile.ZipFile(a.zip).extractall(root)
        D = os.path.dirname(glob.glob(f'{root}/**/train_meta.csv', recursive=True)[0])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '', flush=True)
    tr = pd.read_csv(f'{D}/train_meta.csv'); te = pd.read_csv(f'{D}/test_meta.csv')
    y = tr.label.map({c: i for i, c in enumerate(CLASSES)}).values
    Xtr = np.load(f'{D}/train_cadences.npy', mmap_mode='r'); Xte = np.load(f'{D}/test_cadences.npy', mmap_mode='r')
    t0 = time.time()
    delta = bandpass_profile(Xte) - bandpass_profile(Xtr)
    Ic = build(Xtr); Ia = build(Xtr, 2, delta); It = build(Xte)
    print(f'inputs built {time.time() - t0:.0f}s', flush=True)
    gtr = torch.tensor(tr.slew_gap_rows.values, device=dev); gte = torch.tensor(te.slew_gap_rows.values, device=dev)
    Tc = torch.tensor(Ic, device=dev); Ta = torch.tensor(Ia, device=dev); Tt = torch.tensor(It, device=dev)
    Y = torch.tensor(y, device=dev)
    tri, hoi = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[0]

    def layout(x, g):
        """(B,12,16,128) + gaps -> (B,4,128,128): panels at global-time rows p*(16+gap)."""
        B = x.shape[0]; out = torch.zeros(B, 4, 128, 128, device=x.device, dtype=torch.float32)
        for gv in (3, 4, 5, 6):
            m = (g == gv).nonzero(as_tuple=True)[0]
            if len(m) == 0: continue
            for p in range(6):
                r0 = p * (16 + gv)
                out[m, 0, r0:r0 + 16] = x[m, p].float(); out[m, 1, r0:r0 + 16] = x[m, 6 + p].float()
                out[m, 2, r0:r0 + 16] = 1.0 if p % 2 == 0 else -1.0; out[m, 3, r0:r0 + 16] = 1.0
        return out

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                             nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(s, w=(32, 64, 128, 192, 256)):
            super().__init__()
            s.b = nn.ModuleList([block(4, w[0])] + [block(w[i - 1], w[i]) for i in range(1, len(w))])
            s.head = nn.Sequential(nn.Linear(2 * w[-1], 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 9))
        def forward(s, x):
            for i, b in enumerate(s.b):
                x = b(x)
                if i < len(s.b) - 1: x = F.max_pool2d(x, 2)
            return s.head(torch.cat([x.amax((2, 3)), x.mean((2, 3))], 1))

    def predict(net, T, idx, g):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            for s in range(0, len(idx), 512):
                j = torch.tensor(idx[s:s + 512], device=dev)
                out.append(torch.softmax(net(layout(T[j], g[j])).float(), 1).cpu().numpy())
        return np.concatenate(out)

    counts = np.bincount(y[tri], minlength=9); w = 1.0 / counts[y[tri]]; w = torch.tensor(w / w.sum(), device=dev)
    tri_t = torch.tensor(tri, device=dev)
    BS = 128; steps = len(tri) // BS
    res = {'hold_c': [], 'hold_a': [], 'test': []}
    for seed in [int(s) for s in a.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * steps, pct_start=0.15)
        scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
        for ep in range(a.epochs):
            net.train(); tl = 0; te0 = time.time()
            for _ in range(steps):
                b = tri_t[torch.multinomial(w, BS, replacement=True)]
                use_a = torch.rand(BS, device=dev) < 0.5
                xb = torch.where(use_a[:, None, None, None], Ta[b], Tc[b])
                xb = layout(xb, gtr[b])
                sh = int(np.random.randint(-6, 7))
                if sh: xb = torch.roll(xb, sh, dims=3)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                    loss = F.cross_entropy(net(xb), Y[b], label_smoothing=0.05)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tl += loss.item()
            msg = f'seed {seed} ep {ep} loss {tl / steps:.4f} {time.time() - te0:.0f}s'
            if ep % 5 == 4 or ep == a.epochs - 1:
                pc = predict(net, Tc, hoi, gtr); pa = predict(net, Ta, hoi, gtr)
                msg += f' | fold0 clean {f1_score(y[hoi], pc.argmax(1), average="macro"):.4f} pks-like {f1_score(y[hoi], pa.argmax(1), average="macro"):.4f}'
            print(msg, flush=True)
        res['hold_c'].append(predict(net, Tc, hoi, gtr)); res['hold_a'].append(predict(net, Ta, hoi, gtr))
        res['test'].append(predict(net, Tt, np.arange(len(It)), gte))
        np.savez(a.out, **{k: np.mean(v, 0) for k, v in res.items()}, hoi=hoi, seeds_done=len(res['test']))
        print(f'saved {a.out} after seed {seed}', flush=True)
    pc = np.mean(res['hold_c'], 0); pa = np.mean(res['hold_a'], 0)
    print(f'FINAL seed-mean fold0 clean {f1_score(y[hoi], pc.argmax(1), average="macro"):.4f} '
          f'pks-like {f1_score(y[hoi], pa.argmax(1), average="macro"):.4f}  total {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
