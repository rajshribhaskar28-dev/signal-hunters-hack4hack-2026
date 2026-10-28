"""Retrain HGB with pseudo-labels on test cadences where two independent models agree
(user-supplied submission_v2.csv, LB 0.8571, and our v3)."""
import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
from train_pipeline import feature_sets, make_model
import warnings; warnings.filterwarnings('ignore')
SET=sys.argv[1]
F=pd.read_pickle('cache/F_train_v5.pkl'); A=[pd.read_pickle(f'cache/F_trainaug{s}_v5.pkl') for s in (2,3)]; T=pd.read_pickle('cache/F_test_v5.pkl')
cols=feature_sets(list(F.columns))[SET]
a=pd.read_csv('/mnt/user-data/attachments/52491fa9-e946-55d9-84b3-5f2ccd6e5936/716c67d2-submission_v2.csv'); b=pd.read_csv('submission_final_v3.csv')
agree=(a.label==b.label).values; pl=a.label.map({c:i for i,c in enumerate(CLASSES)}).values
print(SET,'agree n',agree.sum(),flush=True)
X=pd.concat([F[cols]]+[x[cols] for x in A]+[T[cols][agree]]*2); Y=np.r_[y,y,y,pl[agree],pl[agree]]
m=make_model(0); m.fit(X,Y); P=m.predict_proba(T[cols]); np.save(f'cache/test_proba_agree_{SET}.npy',P); print(SET,'done',flush=True)
