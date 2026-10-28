"""Minimal numpy CNN (channels-last, im2col + BLAS matmul) with manual backprop and Adam."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

class Conv:
    def __init__(self, cin, cout, kh, kw, rng):
        self.kh, self.kw, self.cin, self.cout = kh, kw, cin, cout
        self.W = (rng.standard_normal((cin * kh * kw, cout)) * np.sqrt(2.0 / (cin * kh * kw))).astype(np.float32)
        self.b = np.zeros(cout, np.float32)
        self.params = [self.W, self.b]
    def fwd(self, x, need_dx=True):
        B, H, Wd, C = x.shape; ph, pw = self.kh // 2, self.kw // 2
        xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
        cols = swv(xp, (self.kh, self.kw), axis=(1, 2))            # (B,H,W,C,kh,kw)
        cols = np.ascontiguousarray(cols).reshape(B * H * Wd, C * self.kh * self.kw)
        self.cache = (cols, x.shape, need_dx)
        return (cols @ self.W + self.b).reshape(B, H, Wd, self.cout)
    def bwd(self, dout):
        cols, shp, need_dx = self.cache; B, H, Wd, C = shp
        d = dout.reshape(-1, self.cout)
        self.gW = cols.T @ d; self.gb = d.sum(0)
        if not need_dx: return None
        dcols = (d @ self.W.T).reshape(B, H, Wd, C, self.kh, self.kw)
        ph, pw = self.kh // 2, self.kw // 2
        dxp = np.zeros((B, H + 2 * ph, Wd + 2 * pw, C), np.float32)
        for i in range(self.kh):
            for j in range(self.kw):
                dxp[:, i:i + H, j:j + Wd, :] += dcols[..., i, j]
        return dxp[:, ph:ph + H, pw:pw + Wd, :]
    def grads(self): return [self.gW, self.gb]

class Relu:
    params = []
    def fwd(self, x): self.m = x > 0; return x * self.m
    def bwd(self, d): return d * self.m
    def grads(self): return []

class Pool:
    params = []
    def __init__(self, ph, pw): self.ph, self.pw = ph, pw
    def fwd(self, x):
        B, H, W, C = x.shape
        r = x.reshape(B, H // self.ph, self.ph, W // self.pw, self.pw, C)
        o = r.max(axis=(2, 4)); self.mask = (r == o[:, :, None, :, None, :]); self.shp = x.shape
        return o
    def bwd(self, d):
        return (self.mask * d[:, :, None, :, None, :]).reshape(self.shp)
    def grads(self): return []

class GlobalPool:  # concat(max, mean) over H,W
    params = []
    def fwd(self, x):
        B, H, W, C = x.shape; f = x.reshape(B, H * W, C)
        mx = f.max(1); self.mask = (f == mx[:, None, :]); self.hw = H * W; self.shp = x.shape
        self.cnt = self.mask.sum(1, keepdims=True)
        return np.concatenate([mx, f.mean(1)], 1)
    def bwd(self, d):
        C = d.shape[1] // 2
        g = self.mask * (d[:, None, :C] / self.cnt) + d[:, None, C:] / self.hw
        return g.reshape(self.shp)
    def grads(self): return []

class Dense:
    def __init__(self, i, o, rng):
        self.W = (rng.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32); self.b = np.zeros(o, np.float32)
        self.params = [self.W, self.b]
    def fwd(self, x): self.x = x; return x @ self.W + self.b
    def bwd(self, d): self.gW = self.x.T @ d; self.gb = d.sum(0); return d @ self.W.T
    def grads(self): return [self.gW, self.gb]

class Net:
    def __init__(self, cin, ncls, rng, width=(32, 64, 96, 128)):
        a, b, c, e = width
        self.c1 = Conv(cin, a, 3, 5, rng)
        self.layers = [self.c1, Relu(), Pool(1, 2),
                       Conv(a, b, 3, 5, rng), Relu(), Pool(2, 2),
                       Conv(b, c, 3, 3, rng), Relu(), Pool(2, 2),
                       Conv(c, e, 3, 3, rng), Relu(), GlobalPool(),
                       Dense(2 * e, 128, rng), Relu(), Dense(128, ncls, rng)]
    def fwd(self, x):
        for i, l in enumerate(self.layers):
            x = l.fwd(x, need_dx=False) if l is self.c1 else l.fwd(x)
        return x
    def bwd(self, d):
        for l in reversed(self.layers):
            d = l.bwd(d)
            if l is self.c1: break
    def params(self): return [p for l in self.layers for p in l.params]
    def grads(self): return [g for l in self.layers for g in l.grads()]

def softmax_ce(logits, y):
    z = logits - logits.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
    n = len(y); loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(n), y] -= 1; return loss, d / n, p

class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p = params; self.lr = lr; self.b1, self.b2, self.eps, self.wd = b1, b2, eps, wd
        self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]; self.t = 0
    def step(self, grads):
        self.t += 1; b1, b2 = self.b1, self.b2
        lr_t = self.lr * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.wd and p.ndim > 1: g = g + self.wd * p
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            p -= (lr_t * m / (np.sqrt(v) + self.eps)).astype(np.float32)

class Net2:
    """Global-time CNN: input (B,128,128,C) = whole cadence on the global time axis."""
    def __init__(self, cin, ncls, rng, width=(16, 32, 64, 96, 128)):
        L = []; c = cin
        for i, w in enumerate(width):
            conv = Conv(c, w, 3, 3, rng); L += [conv, Relu()]
            if i < len(width) - 1: L.append(Pool(2, 2))
            c = w
        self.c1 = L[0]
        self.layers = L + [GlobalPool(), Dense(2 * c, 128, rng), Relu(), Dense(128, ncls, rng)]
    fwd = Net.fwd; bwd = Net.bwd; params = Net.params; grads = Net.grads
