import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
import warnings; warnings.filterwarnings('ignore')
ver=sys.argv[1]; mode=sys.argv[2]   # mode: clean | mix
drop_cols=sys.argv[3].split(',') if len(sys.argv)>3 and sys.argv[3] else []
F=pd.read_pickle(f'cache/F_train_{ver}.pkl'); AUG=__import__('os').environ.get('AUG','1'); Fa=pd.read_pickle(f'cache/F_trainaug{AUG}_{ver}.pkl')
cols=[c for c in F.columns if c.endswith('_diff') or not any(c.startswith(d) for d in drop_cols)]
F=F[cols]; Fa=Fa[cols]
skf=StratifiedKFold(5,shuffle=True,random_state=42)
Pc=np.zeros((len(F),9)); Pa=np.zeros((len(F),9)); t=time.time()
for k,(a,b) in enumerate(skf.split(F,y)):
    m=make_hgb(learning_rate=0.1)
    if mode=='clean': m.fit(F.iloc[a],y[a])
    else: m.fit(pd.concat([F.iloc[a],Fa.iloc[a]]),np.concatenate([y[a],y[a]]))
    Pc[b]=m.predict_proba(F.iloc[b]); Pa[b]=m.predict_proba(Fa.iloc[b])
tag=f'{ver}_{mode}_aug{AUG}'+('_drop' if drop_cols else '')
np.save(f'cache/oofc_{tag}.npy',Pc); np.save(f'cache/oofa_{tag}.npy',Pa)
report(Pc,f'{tag} -> clean holdout'); report(Pa,f'{tag} -> PKS-aug holdout'); print('time',time.time()-t, 'nfeat',len(cols))
