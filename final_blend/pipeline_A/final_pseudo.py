"""Self-training stage: HGB retrained with confident pseudo-labels on the real (unlabeled) PKS test cadences.
Pseudo-labels come from the HGB+CNN blend (submission_final_v2_proba.npy); tau=0.6 chosen on the simulated-shift holdout."""
import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
from train_pipeline import feature_sets, make_model
import warnings; warnings.filterwarnings('ignore')
SET=sys.argv[1]; SEEDS=[int(s) for s in sys.argv[2].split(',')]; TAU=0.6
F=pd.read_pickle('cache/F_train_v5.pkl'); A=[pd.read_pickle(f'cache/F_trainaug{s}_v5.pkl') for s in (2,3)]; T=pd.read_pickle('cache/F_test_v5.pkl')
cols=feature_sets(list(F.columns))[SET]
B=np.load('submission_final_v2_proba.npy'); conf=B.max(1)>=TAU; pl=B.argmax(1)
print(SET,'pseudo n',conf.sum(),flush=True)
X=pd.concat([F[cols]]+[a[cols] for a in A]+[T[cols][conf]]); Y=np.r_[y,y,y,pl[conf]]
P=np.zeros((len(T),9))
for s in SEEDS:
    m=make_model(s); m.fit(X,Y); P+=m.predict_proba(T[cols])/len(SEEDS); print(SET,'seed',s,'done',flush=True)
np.save(f'cache/test_proba_pseudo_{SET}.npy',P)
