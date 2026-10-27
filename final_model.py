"""Final model: fit / validate / predict, shared by experiments and train_pipeline.py."""
import time, pickle, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from train import CLASSES, D, EXCLUDE, make_model, report, cm

GROUPS = {'1': 'feat', '2': 'feat2', '3': 'feat3', '4': 'feat4', '5': 'feat5'}


SELECT = None
# pixel-extreme statistics: strongly telescope dependent (KS~0.3 GBT vs PKS) and redundant
# with the null-calibrated detectors -> excluded
EXCLUDE_PREFIX = ('zmax_', 'frac3_', 'frac5_', 'pos_mean_')


def feats(tag, groups):
    parts = []
    for g in groups:
        f = pd.read_pickle(f'cache/{GROUPS[g]}_{tag}.pkl')
        parts.append(f.drop(columns=[c for c in f.columns if c in EXCLUDE or c.startswith(EXCLUDE_PREFIX)]))
    X = pd.concat(parts, axis=1)
    return X[SELECT] if SELECT is not None else X


def labels():
    tr = pd.read_csv(f'{D}/train_meta.csv')
    return np.array([CLASSES.index(v) for v in tr.label])


def apply_weights(P, w):
    return P * np.asarray(w)[None, :]


def validate(cfg, fold=0, name=None, log=True):
    """Train on fold-train rows (clean + augmented copies of the same rows), score on the
    held-out rows both clean and PKS-like perturbed (aug seed cfg['val_aug'])."""
    global SELECT
    yi = labels(); g = cfg['groups']
    tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yi)), yi))[fold]
    if cfg.get('select_in_fold'):
        select_features(cfg, rows=tri)        # selection sees only this fold's training rows
    Xc = feats('train', g)
    Xtr = [Xc.iloc[tri]] + [feats(f'train_aug{a}', g).iloc[tri] for a in cfg['augs']]
    ytr = np.concatenate([yi[tri]] * len(Xtr))
    t = time.time()
    m = make_model('hgb', cfg.get('seed', 0), **cfg['hgb']).fit(pd.concat(Xtr), ytr)
    rt = round(time.time() - t)
    out = {}
    for vname, Xv in [('cleanval', Xc), ('pksval', feats(f"train_aug{cfg['val_aug']}", g))]:
        P = m.predict_proba(Xv.iloc[vai])
        np.save(f'cache/P_{name}_{vname}_f{fold}.npy', P)
        out[vname] = P
        if log:
            report(f'{name}_{vname}_f{fold}', yi[vai], P, {'model': f"HGB {cfg['hgb']}",
                   'features': f"groups {g} ({Xc.shape[1]})", 'validation': f'strat fold{fold}/5 {vname}',
                   'seed': cfg.get('seed', 0), 'runtime_s': rt})
    np.save(f'cache/va_f{fold}.npy', vai)
    return out, yi[vai]


def select_features(cfg, rows=None):
    """ExtraTrees impurity ranking on training rows (clean + augmented copies); top-K."""
    from sklearn.ensemble import ExtraTreesClassifier
    global SELECT
    SELECT = None
    yi = labels(); g = cfg['groups']
    rows = np.arange(len(yi)) if rows is None else rows
    Xs = [feats('train', g).iloc[rows]] + [feats(f'train_aug{a}', g).iloc[rows] for a in cfg['augs']]
    et = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', min_samples_leaf=3, class_weight='balanced',
                              n_jobs=2, random_state=0).fit(pd.concat(Xs), np.concatenate([yi[rows]] * len(Xs)))
    imp = pd.Series(et.feature_importances_, index=Xs[0].columns).sort_values(ascending=False)
    imp.to_csv('feature_importance_final.csv')
    SELECT = list(imp.index[:cfg['n_features']])
    return SELECT


def fit_predict(cfg, out='submission.csv', tag='final'):
    import json
    select_features(cfg)
    json.dump(SELECT, open('selected_features_final.json', 'w'))
    yi = labels(); g = cfg['groups']
    Xs = [feats('train', g)] + [feats(f'train_aug{a}', g) for a in cfg['augs']]
    X = pd.concat(Xs); y = np.concatenate([yi] * len(Xs))
    Xt = feats('test', g)[X.columns]
    def fit_avg(Xf, yf):
        Pa = np.zeros((len(Xt), len(CLASSES)))
        for s in cfg.get('seeds', [0]):
            m = make_model('hgb', s, **cfg['hgb']).fit(Xf, yf)
            Pa += m.predict_proba(Xt) / len(cfg.get('seeds', [0]))
        return Pa
    P = fit_avg(X, y)
    # optional self-training on the unlabeled PKS test cadences (no test labels are used):
    # confident predictions become pseudo-labels, the model is refit, repeated pl_rounds times
    for r in range(cfg.get('pl_rounds', 0)):
        sel = P.max(1) >= cfg['pl_thr']
        print(f'pseudo-label round {r+1}: {sel.sum()} test rows', flush=True)
        P = fit_avg(pd.concat([X, Xt[sel]]), np.concatenate([y, P.argmax(1)[sel]]))
    np.save(f'cache/testP_{tag}.npy', P)
    Pw = apply_weights(P, cfg.get('class_weights', [1] * len(CLASSES)))
    te = pd.read_csv(f'{D}/test_meta.csv'); ss = pd.read_csv(f'{D}/sample_submission.csv')
    sub = pd.DataFrame({'id': te.id, 'label': [CLASSES[i] for i in Pw.argmax(1)]})
    sub = ss[['id']].merge(sub, on='id', how='left')
    assert len(sub) == 10000 and sub.label.notna().all() and sub.id.is_unique
    sub.to_csv(out, index=False)
    print(sub.label.value_counts())
    return P
