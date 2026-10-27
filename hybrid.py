"""Hybrid network (numpy): CNN branch on the six panels + MLP branch on physics features.

CNN branch : [conv -> ReLU -> maxpool2] x3 -> conv -> ReLU -> global avg+max   (2*C3 units)
Feature branch : dense(Fd -> 128) -> ReLU
Head : concat -> dense(128) -> ReLU -> dense(9)
Forward/backward by hand; reuses Conv / pooling from cnn.py.
"""
import numpy as np
from cnn import Conv, relu_f, pool_f, pool_b, softmax, loss_grad, F


class Dense:
    def __init__(self, nin, nout, rng, scale=None):
        s = np.sqrt(2.0 / nin) if scale is None else scale
        self.W = (rng.standard_normal((nin, nout)) * s).astype(F)
        self.b = np.zeros(nout, F)

    def fwd(self, x):
        self.x = x
        return x @ self.W + self.b

    def bwd(self, g):
        self.dW = self.x.T @ g; self.db = g.sum(0)
        return g @ self.W.T


class HybridNet:
    def __init__(self, nfeat, cin=6, widths=(24, 48, 64, 96), kw=(5, 3, 3, 3), hf=128, hh=128, ncls=9, seed=0):
        rng = np.random.default_rng(seed)
        self.convs = []; c = cin
        for i, w in enumerate(widths):
            self.convs.append(Conv(c, w, 3, kw[i], rng)); c = w
        self.npool = len(widths) - 1
        self.fd = Dense(nfeat, hf, rng)
        self.h = Dense(2 * c + hf, hh, rng)
        self.o = Dense(hh, ncls, rng, scale=np.sqrt(1.0 / hh))

    def forward(self, x, f):
        cache = []; h = x
        for i, cv in enumerate(self.convs):
            a = cv.fwd(h); r = relu_f(a)
            if i < self.npool:
                p, mask = pool_f(r); cache.append((a, mask)); h = p
            else:
                cache.append((a, None)); h = r
        self.hlast = h; self.mx = h.max((1, 2))
        cf = np.concatenate([h.mean((1, 2)), self.mx], 1)
        self.af = self.fd.fwd(f); hf = relu_f(self.af)
        z = np.concatenate([cf, hf], 1)
        self.ah = self.h.fwd(z); hh = relu_f(self.ah)
        self.cache = cache; self.ncf = cf.shape[1]
        return self.o.fwd(hh)

    def backward(self, g):
        g = self.o.bwd(g) * (self.ah > 0)
        gz = self.h.bwd(g)
        gcf, ghf = gz[:, :self.ncf], gz[:, self.ncf:]
        self.fd.bwd(ghf * (self.af > 0))
        h = self.hlast; B, H, W, C = h.shape
        gavg, gmx = gcf[:, :C], gcf[:, C:]
        mm = (h == self.mx[:, None, None, :])
        gh = gavg[:, None, None, :] / (H * W) + mm * gmx[:, None, None, :] / np.maximum(mm.sum((1, 2), keepdims=True), 1)
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
        for d in (self.fd, self.h, self.o):
            out += [(d, 'W', 'dW'), (d, 'b', 'db')]
        return out


def grad_check():
    rng = np.random.default_rng(3)
    net = HybridNet(nfeat=7, cin=2, widths=(3, 4, 5), kw=(5, 3, 3), hf=6, hh=5, ncls=3, seed=3)
    for o, p, _ in net.plist():
        setattr(o, p, getattr(o, p).astype(np.float64))
    x = rng.standard_normal((2, 8, 16, 2)); f = rng.standard_normal((2, 7)); y = np.array([1, 2]); cw = np.ones(3)
    l, g = loss_grad(net.forward(x, f), y, cw); net.backward(g.astype(np.float64))
    worst = 0
    for o, p, gn in net.plist():
        W = getattr(o, p); G = getattr(o, gn)
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in W.shape); old = W[idx]; e = 1e-5
            W[idx] = old + e; l1, _ = loss_grad(net.forward(x, f), y, cw)
            W[idx] = old - e; l2, _ = loss_grad(net.forward(x, f), y, cw); W[idx] = old
            num = (l1 - l2) / (2 * e); worst = max(worst, abs(num - G[idx]) / max(1e-7, abs(num) + abs(G[idx])))
    print('hybrid grad check max relative error', worst)


if __name__ == '__main__':
    grad_check()
