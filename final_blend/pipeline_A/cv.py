import numpy as np, pandas as pd, time, json, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
import os; D=os.environ.get('SH_DATA','data')
tr=pd.read_csv(f'{D}/train_meta.csv')
CLASSES=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
y=tr.label.map({c:i for i,c in enumerate(CLASSES)}).values
def make_hgb(seed=0,**kw):
    p=dict(max_iter=500,learning_rate=0.06,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=1.0,
           class_weight='balanced',early_stopping=True,validation_fraction=0.1,n_iter_no_change=30,random_state=seed)
    p.update(kw); return HistGradientBoostingClassifier(**p)
def oof(F, model_fn, folds=5, seed=42):
    skf=StratifiedKFold(folds,shuffle=True,random_state=seed)
    P=np.zeros((len(F),len(CLASSES)))
    for k,(a,b) in enumerate(skf.split(F,y)):
        m=model_fn(); m.fit(F.iloc[a],y[a]); P[b]=m.predict_proba(F.iloc[b])
    return P
def report(P,name,log=True):
    pred=P.argmax(1); mf=f1_score(y,pred,average='macro'); pc=f1_score(y,pred,average=None)
    print(f'{name}: macroF1={mf:.4f}',' '.join(f'{c[:6]}={v:.3f}' for c,v in zip(CLASSES,pc)),flush=True)
    return mf,pc
