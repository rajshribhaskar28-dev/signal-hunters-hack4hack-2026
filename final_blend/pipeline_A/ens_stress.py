import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
import warnings; warnings.filterwarnings('ignore')
from train_pipeline import feature_sets
VER=sys.argv[1] if len(sys.argv)>1 else 'v5'
F=pd.read_pickle(f'cache/F_train_{VER}.pkl'); Fa=pd.read_pickle(f'cache/F_trainaug2_{VER}.pkl')
sets=feature_sets(list(F.columns))
skf=StratifiedKFold(5,shuffle=True,random_state=42)
out={k:(np.zeros((len(F),9)),np.zeros((len(F),9))) for k in sets}
for k,(a,b) in enumerate(skf.split(F,y)):
    for nm,cols in sets.items():
        m=make_hgb(learning_rate=0.1); m.fit(pd.concat([F.iloc[a][cols],Fa.iloc[a][cols]]),np.r_[y[a],y[a]])
        out[nm][0][b]=m.predict_proba(F.iloc[b][cols]); out[nm][1][b]=m.predict_proba(Fa.iloc[b][cols])
    print('fold',k,flush=True)
for nm in sets: np.save(f'cache/ens_{VER}_{nm}_c.npy',out[nm][0]); np.save(f'cache/ens_{VER}_{nm}_a.npy',out[nm][1])
for i,d in enumerate(['clean','aug']):
    for nm in sets: report(out[nm][i],f'{nm} {d}')
    report(sum(out[k][i] for k in sets)/len(sets),f'ENS {d}')
