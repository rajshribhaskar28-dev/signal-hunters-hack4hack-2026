"""Compact numpy CNN (no DL framework available in this environment).
Input: preprocessed z cadence, the 6 panels as channels, NHWC (B,16,128,6).
Architecture: [conv3x3 -> ReLU -> maxpool2] x3 -> conv3x3 -> ReLU -> global(avg,max) -> linear(9)
Forward/backward written by hand; Adam optimiser; balanced class-weighted cross-entropy.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

F = np.float32


def im2col(x, kh, kw):
    B, H, W, C = x.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    v = swv(xp, (kh, kw), axis=(1, 2))                      # B,H,W,C,kh,kw
    return np.ascontiguousarray(v).reshape(B * H * W, C * kh * kw)


class Conv:
    def __init__(self, cin, cout, kh=3, kw=3, rng=None):
        self.kh, self.kw, self.cin, self.cout = kh, kw, cin, cout
        self.W = (rng.standard_normal((cin * kh * kw, cout)) * np.sqrt(2.0 / (cin * kh * kw))).astype(F)
        self.b = np.zeros(cout, F)

    def fwd(self, x):
        self.xshape = x.shape
        B, H, W, _ = x.shape
        self.cols = im2col(x, self.kh, self.kw)
        return (self.cols @ self.W + self.b).reshape(B, H, W, self.cout)

    def bwd(self, g, need_dx=True):
        B, H, W, _ = self.xshape
        g2 = g.reshape(-1, self.cout)
        self.dW = self.cols.T @ g2
        self.db = g2.sum(0)
        self.cols = None
        if not need_dx:
            return None
        # dX = conv(g, flipped W with in/out swapped)
        Wr = self.W.reshape(self.cin, self.kh, self.kw, self.cout)[:, ::-1, ::-1, :]
        Wt = np.ascontiguousarray(Wr.transpose(3, 1, 2, 0)).reshape(self.cout * self.kh * self.kw, self.cin)
        # im2col ordering is (C, kh, kw): need Wt rows ordered (cout, kh, kw)
        return (im2col(g, self.kh, self.kw) @ Wt).reshape(B, H, W, self.cin)

    def params(self):
        return [(self.W, 'dW'), (self.b, 'db')]


def relu_f(x):
    return np.maximum(x, 0)


def pool_f(x):
    B, H, W, C = x.shape
    r = x.reshape(B, H // 2, 2, W // 2, 2, C)
    m = r.max(axis=(2, 4))
    mask = (r == m[:, :, None, :, None, :])
    return m, mask


def pool_b(g, mask):
    B, h, _, w, _, C = mask.shape
    gg = mask * g[:, :, None, :, None, :]
    # split ties: normalise by count
    cnt = mask.sum(axis=(2, 4), keepdims=True)
    gg = gg / np.maximum(cnt, 1)
    return gg.reshape(B, h * 2, w * 2, C)


class Net:
    def __init__(self, cin=6, widths=(24, 48, 64, 96), ncls=9, seed=0, kw=(5, 3, 3, 3)):
        rng = np.random.default_rng(seed)
        self.convs = []
        c = cin
        for i, w in enumerate(widths):
            self.convs.append(Conv(c, w, 3, kw[i], rng)); c = w
        self.Wd = (rng.standard_normal((2 * c, ncls)) * np.sqrt(1.0 / (2 * c))).astype(F)
        self.bd = np.zeros(ncls, F)
        self.npool = len(widths) - 1

    def forward(self, x, train=False):
        cache = []
        h = x
        for i, cv in enumerate(self.convs):
            a = cv.fwd(h)
            r = relu_f(a)
            if i < self.npool:
                p, mask = pool_f(r)
                cache.append((a, mask)); h = p
            else:
                cache.append((a, None)); h = r
        B, H, W, C = h.shape
        avg = h.mean((1, 2)); mx = h.max((1, 2))
        self.hlast = h; self.mx = mx
        feat = np.concatenate([avg, mx], 1)
        self.feat = feat; self.cache = cache
        return feat @ self.Wd + self.bd

    def backward(self, glogit):
        self.dWd = self.feat.T @ glogit; self.dbd = glogit.sum(0)
        gf = glogit @ self.Wd.T
        h = self.hlast; B, H, W, C = h.shape
        gavg, gmx = gf[:, :C], gf[:, C:]
        mmask = (h == self.mx[:, None, None, :])
        gh = gavg[:, None, None, :] / (H * W) + mmask * gmx[:, None, None, :] / np.maximum(mmask.sum((1, 2), keepdims=True), 1)
        for i in range(len(self.convs) - 1, -1, -1):
            a, mask = self.cache[i]
            if mask is not None:
                gh = pool_b(gh, mask)
            gh = gh * (a > 0)
            gh = self.convs[i].bwd(gh, need_dx=(i > 0))
        self.cache = None

    def plist(self):
        out = []
        for cv in self.convs:
            out += [(cv, 'W', 'dW'), (cv, 'b', 'db')]
        out += [(self, 'Wd', 'dWd'), (self, 'bd', 'dbd')]
        return out


class Adam:
    def __init__(self, net, lr=2e-3, b1=0.9, b2=0.999, wd=1e-4):
        self.net, self.lr, self.b1, self.b2, self.wd, self.t = net, lr, b1, b2, wd, 0
        self.m = [np.zeros_like(getattr(o, p)) for o, p, _ in net.plist()]
        self.v = [np.zeros_like(getattr(o, p)) for o, p, _ in net.plist()]

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for i, (o, p, g) in enumerate(self.net.plist()):
            w = getattr(o, p); gr = getattr(o, g)
            if p in ('W', 'Wd'):
                gr = gr + self.wd * w
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gr
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * gr * gr
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            w -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(F)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def loss_grad(logits, y, cw):
    P = softmax(logits)
    B = len(y)
    w = cw[y]
    loss = -(w * np.log(P[np.arange(B), y] + 1e-9)).sum() / w.sum()
    g = P.copy(); g[np.arange(B), y] -= 1
    g = g * (w / w.sum())[:, None]
    return loss, g.astype(F)


def grad_check():
    rng = np.random.default_rng(1)
    net = Net(cin=2, widths=(3, 4, 5), ncls=3, seed=1, kw=(5, 3, 3))
    x = rng.standard_normal((2, 8, 16, 2)).astype(np.float64)
    y = np.array([0, 2]); cw = np.ones(3)
    for o, p, _ in net.plist():
        setattr(o, p, getattr(o, p).astype(np.float64))
    lg = net.forward(x); l, g = loss_grad(lg, y, cw); net.backward(g.astype(np.float64))
    maxrel = 0
    for o, p, gname in net.plist():
        W = getattr(o, p); G = getattr(o, gname)
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in W.shape)
            old = W[idx]; e = 1e-5
            W[idx] = old + e; l1, _ = loss_grad(net.forward(x), y, cw)
            W[idx] = old - e; l2, _ = loss_grad(net.forward(x), y, cw)
            W[idx] = old
            num = (l1 - l2) / (2 * e)
            rel = abs(num - G[idx]) / max(1e-7, abs(num) + abs(G[idx]))
            maxrel = max(maxrel, rel)
    print('grad check max relative error', maxrel)


if __name__ == '__main__':
    grad_check()
