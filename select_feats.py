"""Rank features by ExtraTrees impurity importance on fold-0 TRAIN rows only (clean + aug2),
so validation rows never influence selection. Writes selected_features.json (top K)."""
import json, sys, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from final_model import feats, labels
K = int(sys.argv[1]) if len(sys.argv) > 1 else 300
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
yi = labels(); Xc = feats('train', '1234'); Xa = feats('train_aug2', '1234')
tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yi))[FOLD]
X = pd.concat([Xc.iloc[tri], Xa.iloc[tri]]); y = np.concatenate([yi[tri], yi[tri]])
et = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=3, class_weight='balanced',
                          n_jobs=2, random_state=0).fit(X, y)
imp = pd.Series(et.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv(f'feature_importance_et_f{FOLD}.csv')
sel = list(imp.index[:K])
json.dump(sel, open(f'selected_features_{K}.json' if FOLD == 0 else f'selected_features_{K}_f{FOLD}.json', 'w'))
print('fold', FOLD, 'top5', list(imp.index[:5]))
print('groups in top K:', pd.Series([c.split('_')[0] for c in sel]).value_counts().head(15).to_dict())
