import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prep import preprocess
from train import CLASSES, D
X = np.load(f'{D}/train_cadences.npy', mmap_mode='r'); tr = pd.read_csv(f'{D}/train_meta.csv')
va = np.load('cache/fold0_va.npy'); P = np.load('cache/P_v1234sel350_aug2_cleanval_f0.npy')
y = np.array([CLASSES.index(v) for v in tr.label])[va]; pr = P.argmax(1)
rows = []
for c in range(9):
    ok = va[(y == c) & (pr == c)]; bad = np.where((y == c) & (pr != c))[0]
    rows.append((ok[0], f'{CLASSES[c]}  OK'))
    if len(bad): rows.append((va[bad[0]], f'{CLASSES[c]} -> {CLASSES[pr[bad[0]]]}'))
fig, ax = plt.subplots(len(rows), 6, figsize=(12, 1.25 * len(rows)))
for i, (k, t) in enumerate(rows):
    z = preprocess(np.asarray(X[k:k + 1]))[0][0]
    for p in range(6):
        ax[i, p].imshow(np.nan_to_num(z[p]), aspect='auto', cmap='magma', vmin=-1.5, vmax=4)
        ax[i, p].set_xticks([]); ax[i, p].set_yticks([])
        if i == 0: ax[i, p].set_title(['ON', 'OFF'][p % 2], fontsize=8)
    ax[i, 0].set_ylabel(t, fontsize=6, rotation=0, ha='right', va='center')
plt.tight_layout(); plt.savefig('error_examples.png', dpi=70)
