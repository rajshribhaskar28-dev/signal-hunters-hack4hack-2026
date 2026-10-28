import os
"""Build CNN input tensors (float16, 12x16x128): z (excess power, clipped) + log-power relative to OFF reference."""
import sys, numpy as np, pandas as pd, time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from prep import preprocess
from augment import pksify
from joblib import Parallel, delayed
D=os.environ.get('SH_DATA','data'); BATCH=200
OFF=[1,3,5]
def inputs(v):
    z,valid,glitch,drop,dev=preprocess(v)
    a=np.clip(z,-3,12)/3.0
    L=v.astype(np.float32)/24.0; L[...,64]=0.5*(L[...,63]+L[...,65]); L[~valid]=np.nan
    spec=np.nanmedian(L,axis=2)                       # (B,6,128)
    ref=np.nanmean(spec[:,OFF],axis=1)                # (B,128)
    r=L-ref[:,None,None,:]
    r=r-np.nanmedian(r[:,OFF],axis=(1,2,3))[:,None,None,None]
    r=np.clip(np.nan_to_num(r),-1,1)*3
    return np.concatenate([a,r],axis=1).astype(np.float16)
def build(name,aug_seed=None):
    X=np.load(f'{D}/{name}_cadences.npy',mmap_mode='r'); delta=np.load('cache/bp_delta.npy')
    def run(s):
        v=np.asarray(X[s:s+BATCH])
        if aug_seed is not None: v=pksify(v,np.random.default_rng(aug_seed*100000+s),delta)
        return inputs(v)
    out=np.lib.format.open_memmap(f'cache/cnn_{name}{"_aug"+str(aug_seed) if aug_seed else ""}.npy',mode='w+',dtype=np.float16,shape=(len(X),12,16,128))
    parts=Parallel(n_jobs=2)(delayed(run)(s) for s in range(0,len(X),BATCH))
    i=0
    for p in parts: out[i:i+len(p)]=p; i+=len(p)
    out.flush()
t=time.time()
build('train'); print('train',time.time()-t,flush=True)
build('train',2); print('aug2',time.time()-t,flush=True)
build('test'); print('test',time.time()-t,flush=True)
