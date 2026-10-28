import numpy as np
from scipy.ndimage import median_filter
def preprocess(v):
    """v: (B,6,16,128) uint8 -> z (B,6,16,128) float32 standardized excess power, valid mask, glitch mask"""
    v=v.astype(np.float32)
    drop=(v==0).all(-1)                      # (B,6,16) dropped rows
    v[...,64]=0.5*(v[...,63]+v[...,65])      # remove GBT centre spike consistently in all data
    v[drop]=np.nan
    L=v/24.0
    # broadband glitch rows: row median deviates strongly from panel median (log2 units)
    rowmed=np.nanmedian(L,axis=-1)           # (B,6,16)
    pmed=np.nanmedian(rowmed,axis=-1,keepdims=True)
    dev=rowmed-pmed
    glitch=np.abs(dev)>0.35
    bad=drop|glitch
    L[bad]=np.nan
    P=np.exp2(L-4)
    bp=np.nanmedian(P,axis=2)                # (B,6,128) per-channel median over time
    bp=np.where(np.isnan(bp),np.nanmedian(bp,axis=-1,keepdims=True),bp)
    bp=median_filter(bp,size=(1,1,9),mode='nearest')
    N=P/bp[:,:,None,:]-1
    med=np.nanmedian(N,axis=(2,3),keepdims=True)
    mad=np.nanmedian(np.abs(N-med),axis=(2,3),keepdims=True)*1.4826+1e-6
    z=(N-med)/mad
    z=np.nan_to_num(z,nan=0.0).astype(np.float32)
    return z,~bad,glitch,drop,dev
