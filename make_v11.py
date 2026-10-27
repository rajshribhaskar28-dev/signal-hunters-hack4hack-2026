"""Reproduce submission A_v11_main from the original competition files.

    python make_v11.py --data /path/to/signal-hunters-round2

Steps (cached in ./cache, skipped when present):
 1. physics feature groups 1,2,3,4,6 for train, test and augmented copies 2 (harsh), 4, 5 (test-matched)
 2. line maps (drift-search maps) for the same sets
 3. HistGradientBoosting (final_config_v10.json, 3 seeds) on clean + 3 augmented copies  -> P_trees
 4. MapNet (1-D CNN on line maps + feature MLP), full data, seeds 0,1,2, 12 epochs         -> P_map
 5. P = 0.8 * P_trees + 0.2 * P_map  -> submission.csv
Single-threaded BLAS (OMP_NUM_THREADS=1) is used for bit-for-bit reproducibility.
"""
import argparse, os, subprocess, sys, json, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(args, env):
    print('>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable] + args, check=True, cwd=HERE, env=env)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.environ.get('SH_DATA', 'data')); a = ap.parse_args()
    env = dict(os.environ, SH_DATA=a.data, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.makedirs(os.path.join(HERE, 'cache'), exist_ok=True)
    sets = [('train', None), ('test', None), ('train', 2), ('train', 4), ('train', 5)]
    for mod, pre in [('features', 'feat'), ('features2', 'feat2'), ('features3', 'feat3'), ('features4', 'feat4'), ('features6', 'feat6')]:
        for split, aug in sets:
            tag = split if aug is None else f'{split}_aug{aug}'
            if not os.path.exists(os.path.join(HERE, 'cache', f'{pre}_{tag}.pkl')):
                sh([f'{mod}.py', split] + ([str(aug)] if aug is not None else []), env)
    for split, aug in sets:
        tag = split if aug is None else f'{split}_aug{aug}'
        if not os.path.exists(os.path.join(HERE, 'cache', f'maps_{tag}.npy')):
            sh(['prep_maps.py', split] + ([str(aug)] if aug is not None else []), env)
    os.environ.update(env); os.chdir(HERE); sys.path.insert(0, HERE)
    from final_model import fit_predict
    from train import CLASSES, D
    cfg = json.load(open('final_config_v10.json'))
    P_trees = fit_predict(cfg, out='submission_trees.csv', tag='v10')
    shutil.copy('selected_features_final.json', 'selected_features_v10.json')
    menv = dict(env, FEAT_GROUPS='12346', SEL_FILE='selected_features_v10.json')
    for s in range(3):
        sh(['train_mapnet.py', f'mf{s}', '12', 'full', str(s)], menv)
    P_map = np.mean([np.load(f'cache/testP_map_mf{s}.npy') for s in range(3)], 0)
    P = 0.8 * P_trees + 0.2 * P_map
    te = pd.read_csv(f'{D}/test_meta.csv'); ss = pd.read_csv(f'{D}/sample_submission.csv')
    sub = ss[['id']].merge(pd.DataFrame({'id': te.id, 'label': np.array(CLASSES)[P.argmax(1)]}), on='id', how='left')
    sub.to_csv('submission.csv', index=False); print(sub.label.value_counts())


if __name__ == '__main__':
    main()
