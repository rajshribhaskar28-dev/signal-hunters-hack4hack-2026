"""Line-map network (numpy): 1-D CNN over frequency on drift-search maps + MLP on physics features.

Input maps M (B, W, C): for each of the 6 panels, per channel position, the best stationary,
positive-slow, negative-slow, fast, steep line significance, the signed drift of the best slow
line, and the time-mean spectrum -> C = 6 x 7 = 42 channels, W = 144 positions.
1-D convs see "which panels are lit at the same channel" directly.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from cnn import softmax, loss_grad, F
from hybrid import Dense


class Conv1:
    def __init__(self, cin, cout, k, rng):
        self.k, self.cin, self.cout = k, cin, cout
        self.W = (rng.standard_normal((cin * k, cout)) * np.sqrt(2.0 / (cin * k))).astype(F)
        self.b = np.zeros(cout, F)

    def _cols(self, x):
        B, Wd, C = x.shape; p = self.k // 2
        xp = np.pad(x, ((0, 0), (p, p), (0, 0)))
        return np.ascontiguousarray(swv(xp, self.k, axis=1)).reshape(B * Wd, C * self.k)   # (C,k) order

    def fwd(self, x):
        self.shape = x.shape; B, Wd, _ = x.shape
        self.cols = self._cols(x)
        return (self.cols @ self.W + self.b).reshape(B, Wd, self.cout)

    def bwd(self, g, need_dx=True):
        B, Wd, _ = self.shape
        g2 = g.reshape(-1, self.cout)
        self.dW = self.cols.T @ g2; self.db = g2.sum(0); self.cols = None
        if not need_dx:
            return None
        Wr = self.W.reshape(self.cin, self.k, self.cout)[:, ::-1, :]
        Wt = np.ascontiguousarray(Wr.transpose(2, 1, 0)).reshape(self.cout * self.k, self.cin)
        return (self._cols(g) @ Wt).reshape(B, Wd, self.cin)


def pool1_f(x):
    B, Wd, C = x.shape
    r = x.reshape(B, Wd // 2, 2, C); m = r.max(2)
    return m, (r == m[:, :, None, :])


def pool1_b(g, mask):
    gg = mask * g[:, :, None, :]; gg = gg / np.maximum(mask.sum(2, keepdims=True), 1)
    B, w, _, C = mask.shape
    return gg.reshape(B, w * 2, C)


class MapNet:
    def __init__(self, cin, nfeat, widths=(64, 64, 96, 128), ks=(5, 5, 5, 3), hf=128, hh=128, ncls=9, seed=0):
        rng = np.random.default_rng(seed)
        self.convs = []; c = cin
        for w, k in zip(widths, ks):
            self.convs.append(Conv1(c, w, k, rng)); c = w
        self.pool_after = {1, 2}
        self.fd = Dense(nfeat, hf, rng); self.h = Dense(2 * c + hf, hh, rng)
        self.o = Dense(hh, ncls, rng, scale=np.sqrt(1.0 / hh))

    def forward(self, x, f):
        cache = []; h = x
        for i, cv in enumerate(self.convs):
            a = cv.fwd(h); r = np.maximum(a, 0)
            if i in self.pool_after:
                p, m = pool1_f(r); cache.append((a, m)); h = p
            else:
                cache.append((a, None)); h = r
        self.hlast = h; self.mx = h.max(1)
        cf = np.concatenate([h.mean(1), self.mx], 1)
        self.af = self.fd.fwd(f); hf = np.maximum(self.af, 0)
        self.ah = self.h.fwd(np.concatenate([cf, hf], 1)); hh = np.maximum(self.ah, 0)
        self.cache = cache; self.ncf = cf.shape[1]
        return self.o.fwd(hh)

    def backward(self, g):
        g = self.o.bwd(g) * (self.ah > 0)
        gz = self.h.bwd(g); gcf, ghf = gz[:, :self.ncf], gz[:, self.ncf:]
        self.fd.bwd(ghf * (self.af > 0))
        h = self.hlast; B, Wd, C = h.shape
        mm = (h == self.mx[:, None, :])
        gh = gcf[:, None, :C] / Wd + mm * gcf[:, None, C:] / np.maximum(mm.sum(1, keepdims=True), 1)
        for i in range(len(self.convs) - 1, -1, -1):
            a, m = self.cache[i]
            if m is not None:
                gh = pool1_b(gh, m)
            gh = gh * (a > 0)
            gh = self.convs[i].bwd(gh, need_dx=(i > 0))
        self.cache = None

    def plist(self):
        out = []
        for cv in self.convs:
            out += [(cv, 'W', 'dW'), (cv, 'b', 'db')]
        for d in (self.fd, self.h, self.o):
            out += [(d, 'W', 'dW'), (d, 'b', 'db')]
        return out


def grad_check():
    rng = np.random.default_rng(5)
    net = MapNet(cin=3, nfeat=4, widths=(4, 5, 6, 7), ks=(3, 3, 3, 3), hf=5, hh=6, ncls=3, seed=5)
    for o, p, _ in net.plist():
        setattr(o, p, getattr(o, p).astype(np.float64))
    x = rng.standard_normal((2, 16, 3)); f = rng.standard_normal((2, 4)); y = np.array([0, 2]); cw = np.ones(3)
    l, g = loss_grad(net.forward(x, f), y, cw); net.backward(g.astype(np.float64))
    worst = 0
    for o, p, gn in net.plist():
        W = getattr(o, p); G = getattr(o, gn)
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in W.shape); old = W[idx]; e = 1e-5
            W[idx] = old + e; l1, _ = loss_grad(net.forward(x, f), y, cw)
            W[idx] = old - e; l2, _ = loss_grad(net.forward(x, f), y, cw); W[idx] = old
            num = (l1 - l2) / (2 * e); worst = max(worst, abs(num - G[idx]) / max(1e-7, abs(num) + abs(G[idx])))
    print('mapnet grad check max relative error', worst)


if __name__ == '__main__':
    grad_check()
