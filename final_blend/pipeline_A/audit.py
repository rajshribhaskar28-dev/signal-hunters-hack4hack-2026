import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
ver=sys.argv[1]
F=pd.read_pickle(f'cache/F_train_{ver}.pkl'); AUG=__import__('os').environ.get('AUG','1'); Fa=pd.read_pickle(f'cache/F_trainaug{AUG}_{ver}.pkl'); T=pd.read_pickle(f'cache/F_test_{ver}.pkl')
rows=[]
for c in F.columns:
    rows.append((c,ks_2samp(F[c],T[c]).statistic,ks_2samp(Fa[c],T[c]).statistic))
K=pd.DataFrame(rows,columns=['feat','ks_clean','ks_aug']).sort_values('ks_clean',ascending=False)
K.to_csv('cache/ks_audit.csv',index=False)
print(K.head(25).round(3).to_string()); print('median KS clean',K.ks_clean.median().round(3),'aug',K.ks_aug.median().round(3))
rng=np.random.default_rng(0); i=rng.choice(len(F),6000,replace=False); j=rng.choice(len(T),6000,replace=False)
for name,S in [('clean',F),('aug',Fa)]:
    Xa=pd.concat([S.iloc[i],T.iloc[j]]); ya=np.r_[np.zeros(6000),np.ones(6000)]
    p=cross_val_predict(HistGradientBoostingClassifier(max_iter=100,random_state=0),Xa,ya,cv=3,method='predict_proba')[:,1]
    print('adversarial AUC train-%s vs test: %.3f'%(name,roc_auc_score(ya,p)))
