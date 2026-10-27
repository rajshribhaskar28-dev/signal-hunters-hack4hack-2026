"""Train (GBT) vs test (PKS) feature distribution audit using unlabeled test features."""
import sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp
a = pd.read_pickle('cache/feat_train.pkl'); b = pd.read_pickle('cache/feat_test.pkl')
c = pd.read_pickle(sys.argv[1]) if len(sys.argv) > 1 else None
rows = []
for col in a.columns:
    r = {'feature': col, 'ks_train_test': ks_2samp(a[col], b[col]).statistic,
         'med_train': a[col].median(), 'med_test': b[col].median(),
         'q95_train': a[col].quantile(.95), 'q95_test': b[col].quantile(.95)}
    if c is not None:
        r['ks_aug_test'] = ks_2samp(c[col], b[col]).statistic
        r['med_aug'] = c[col].median()
    rows.append(r)
R = pd.DataFrame(rows).sort_values('ks_train_test', ascending=False)
R.to_csv('audit_shift.csv', index=False)
pd.set_option('display.width', 250)
print(R.head(40).round(3).to_string())
if c is not None:
    print('mean KS train-test', R.ks_train_test.mean().round(3), 'aug-test', R.ks_aug_test.mean().round(3))
