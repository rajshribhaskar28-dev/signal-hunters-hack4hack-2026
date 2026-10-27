import time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from train import *
tr = pd.read_csv(f'{D}/train_meta.csv'); y = tr.label.values
yi = np.array([CLASSES.index(v) for v in y])
F = load('train').drop(columns=EXCLUDE)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
tri, vai = next(skf.split(F, yi))
t = time.time()
m = make_model('hgb', 0, max_iter=150, lr=0.12).fit(F.iloc[tri], yi[tri])
P = m.predict_proba(F.iloc[vai])
report('hgb_v1_fold0_fast', yi[vai], P, {'model': 'HGB it150 lr.12', 'features': f'physics_v1 ({F.shape[1]})',
       'validation': 'stratified fold0/5', 'seed': 0, 'runtime_s': round(time.time() - t)})
print(cm(yi[vai], P)); np.save('cache/fold0_P_v1.npy', P); np.save('cache/fold0_va.npy', vai)
print('fit s', time.time() - t)
