import sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
import os; D=os.environ.get('SH_DATA','data')
X=np.load(f'{D}/train_cadences.npy',mmap_mode='r'); T=np.load(f'{D}/test_cadences.npy',mmap_mode='r')
tr=pd.read_csv(f'{D}/train_meta.csv')
def flat(c):
    c=c.astype(np.float32); m=c==0; c[m]=np.nan
    med=np.nanmedian(c,axis=1,keepdims=True)  # per panel per channel over time
    c=c-med
    c=c-np.nanmedian(c,axis=(1,2),keepdims=True)
    return np.nan_to_num(c)
labels=sorted(tr.label.unique()); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
rng=np.random.default_rng(seed)
for lab in labels:
    idx=rng.choice(np.where(tr.label==lab)[0],4,replace=False)
    fig,ax=plt.subplots(4,6,figsize=(14,6))
    for r,i in enumerate(idx):
        c=flat(X[i])
        for p in range(6):
            ax[r,p].imshow(c[p],aspect='auto',cmap='magma',vmin=-10,vmax=40); ax[r,p].set_xticks([]);ax[r,p].set_yticks([])
        ax[r,0].set_ylabel(f'{i} g{tr.slew_gap_rows[i]}',fontsize=7)
    fig.suptitle(lab); plt.tight_layout(); plt.savefig(f'plots/{lab}_{seed}.png',dpi=60); plt.close()
fig,ax=plt.subplots(6,6,figsize=(14,9))
for r,i in enumerate(rng.choice(10000,6,replace=False)):
    c=flat(T[i])
    for p in range(6): ax[r,p].imshow(c[p],aspect='auto',cmap='magma',vmin=-10,vmax=40);ax[r,p].set_xticks([]);ax[r,p].set_yticks([])
plt.tight_layout(); plt.savefig(f'plots/test_{seed}.png',dpi=60)
# raw bandpass compare
fig,ax=plt.subplots(1,2,figsize=(12,3))
a=np.asarray(X[:3000]).astype(np.float32);a[a==0]=np.nan; b=np.asarray(T[:3000]).astype(np.float32);b[b==0]=np.nan
ax[0].plot(np.nanmedian(a,axis=(0,1,2)),label='GBT');ax[0].plot(np.nanmedian(b,axis=(0,1,2)),label='PKS');ax[0].legend()
ax[1].plot(np.nanmedian(a,axis=(0,1,3)).ravel(),label='GBT');ax[1].plot(np.nanmedian(b,axis=(0,1,3)).ravel(),label='PKS')
plt.savefig('plots/bandpass.png',dpi=70)
