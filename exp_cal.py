"""Evaluate the calibrated feature set: fold-0 clean / PKS-like validation (in-fold selection)
and adversarial train-vs-test separability."""
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import final_model
from final_model import validate, feats
name = sys.argv[1]
cfg = json.load(open('final_config.json')); cfg['select_in_fold'] = True
if len(sys.argv) > 2: cfg['augs'] = [int(a) for a in sys.argv[2].split(',')] if sys.argv[2] else []
validate(cfg, fold=0, name=name)
sel = final_model.SELECT
T = feats('test', '1234'); rng = np.random.default_rng(0)
for nm, tag in [('clean', 'train'), ('aug2', 'train_aug2')]:
    S = feats(tag, '1234').iloc[rng.choice(18000, 10000, replace=False)]
    X = pd.concat([S, T]); y = np.r_[np.zeros(len(S)), np.ones(len(T))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, max_bins=64, random_state=0), X, y, cv=3,
                          method='predict_proba')[:, 1]
    print(name, nm, 'vs test adversarial AUC', round(roc_auc_score(y, p), 4), flush=True)
