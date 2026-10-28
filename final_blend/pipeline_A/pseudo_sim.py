import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
from train_pipeline import feature_sets
import warnings; warnings.filterwarnings('ignore')
TAU=float(sys.argv[1]); WP=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
F=pd.read_pickle('cache/F_train_v5.pkl'); Fa=pd.read_pickle('cache/F_trainaug2_v5.pkl')
cols=feature_sets(list(F.columns))['v4drop']
tri,hoi=list(StratifiedKFold(5,shuffle=True,random_state=42).split(F,y))[0]
H=((np.load('cache/ens_v5_v3cols_a.npy')+np.load('cache/ens_v5_v4drop_a.npy'))/2)[hoi]
C=np.mean([np.load(f'cache/cnn_s{s}_h0_hold_a.npy') for s in (0,1)],0)
B=0.65*H+0.35*C
conf=B.max(1)>=TAU; pl=B.argmax(1)
print(f'tau {TAU}: pseudo n={conf.sum()} of {len(hoi)}, pseudo-label accuracy={np.mean(pl[conf]==y[hoi][conf]):.4f}',flush=True)
Xtr=pd.concat([F.iloc[tri][cols],Fa.iloc[tri][cols],Fa.iloc[hoi][cols][conf]]); ytr=np.r_[y[tri],y[tri],pl[conf]]
sw=np.r_[np.ones(2*len(tri)),np.full(conf.sum(),WP)]
m=make_hgb(learning_rate=0.1); m.fit(Xtr,ytr,sample_weight=sw)
Pn=m.predict_proba(Fa.iloc[hoi][cols])
base=np.load('cache/ens_v5_v4drop_a.npy')[hoi]
yh=y[hoi]
f=lambda P: f1_score(yh,P.argmax(1),average='macro')
print(f'HGB v4drop only: before {f(base):.4f} after {f(Pn):.4f}')
print(f'blend w/ CNN 0.35: before {f(0.65*base+0.35*C):.4f} after {f(0.65*Pn+0.35*C):.4f}')
np.save(f'cache/pseudo_sim_{TAU}.npy',Pn)
