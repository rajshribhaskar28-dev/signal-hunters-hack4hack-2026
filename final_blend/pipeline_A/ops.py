import numpy as np, scipy.sparse as sp
NR,NC=16,128
def drift_op(drifts, width=1):
    """sparse op: z (16*128) -> line sums for (drift,start channel at row 0). returns op, norm (sqrt count)"""
    rows=[];cols=[];cnt=np.zeros(len(drifts)*NC)
    for i,d in enumerate(drifts):
        for c in range(NC):
            o=i*NC+c
            for r in range(NR):
                cc=int(round(c+d*r))
                if 0<=cc<NC:
                    rows.append(o);cols.append(r*NC+cc);cnt[o]+=1
    A=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(drifts)*NC,NR*NC))
    return A,np.sqrt(np.maximum(cnt,1)).astype(np.float32),cnt
def disp_curve(f0,Dt):
    f=np.arange(NC)+f0
    s=(f**-2.0-f[-1]**-2.0)/(f[0]**-2.0-f[-1]**-2.0)
    return Dt*s   # delay (rows) per channel, 0 at top channel, Dt at channel 0
def disp_op(f0,Dts):
    rows=[];cols=[];meta=[];o=0;cnt=[]
    for Dt in Dts:
        dl=np.round(disp_curve(f0,Dt)).astype(int)
        for t0 in range(-int(np.ceil(Dt)),NR):
            n=0
            for c in range(NC):
                t=t0+dl[c]
                if 0<=t<NR: rows.append(o);cols.append(t*NC+c);n+=1
            meta.append((Dt,t0));cnt.append(n);o+=1
    A=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(o,NR*NC))
    return A,np.array(meta),np.array(cnt)
