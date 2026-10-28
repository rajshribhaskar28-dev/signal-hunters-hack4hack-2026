"""Build the ranked CSV set from: my feature ensemble (M), the other pipeline's GBM (T), my CNNs (C)."""
import numpy as np, pandas as pd, os, json
CL=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
W='/home/claude/work'; R='/home/claude/ext/v11repo'; O='/home/claude/night/out'
D='/home/claude/data/signal-hunters-round2'
ss=pd.read_csv(f'{D}/sample_submission.csv')
M=np.load(f'{W}/cache/test_proba.npy')
Mpl=(np.load(f'{W}/cache/test_proba_agree_v4drop.npy')+np.load(f'{W}/cache/test_proba_agree_v3cols.npy'))/2
T9=np.load(f'{R}/blend/testP_gbm_v9.npy')
V='/home/claude/night/v11/cache'
T11=0.8*np.load(f'{V}/testP_v10.npy')+0.2*np.mean([np.load(f'{V}/testP_map_mf{s}.npy') for s in range(3)],0)
T=T11
cnn_files=['gnp_s2_h0+gnp_s3_h0_pl','np_s0_h0','cnn_s0_h0+cnn_s1_h0']+[f for f in os.environ.get('EXTRA_CNN','').split(',') if f]
def ld(f): return np.mean([np.load(f'{W}/cache/{g}_test.npy') for g in f.split('+')],0)
C=np.mean([ld(f) for f in cnn_files],0)
v11=pd.read_csv(f'{R}/submission.csv'); v2=pd.read_csv('/mnt/user-data/attachments/52491fa9-e946-55d9-84b3-5f2ccd6e5936/716c67d2-submission_v2.csv')
assert (v11.id==ss.id).all() and (v2.id==ss.id).all() and T.shape==(10000,9)
def write(name,labels,note):
    s=pd.DataFrame({'id':ss.id.values,'label':labels})
    assert list(s.columns)==['id','label'] and len(s)==10000 and s.id.is_unique and s.label.isin(CL).all() and s.label.notna().all()
    s.to_csv(f'{O}/{name}',index=False); return s
def lab(P): return np.array(CL)[P.argmax(1)]
files=[]
def add(name,P=None,labels=None,note=''):
    L=lab(P) if P is not None else labels; s=write(name,L,note)
    files.append(dict(file=name,note=note,agree_v2=round((s.label==v2.label).mean(),4),agree_v11=round((s.label==v11.label).mean(),4)))
add('01_blend_M35_T35_C30.csv',0.35*M+0.35*T+0.3*C,note='MAIN: mine 0.35 + v11 0.35 + CNNs 0.30 (incl. 2 global-time CNNs)')
add('02_blend_Mpl35_T35_C30.csv',0.35*Mpl+0.35*T+0.3*C,note='as 01 but my HGB retrained with agreement pseudo-labels')
add('03_blend_M35_T9_35_C30.csv',0.35*M+0.35*T9+0.3*C,note='as 01 but second pipeline = v9 GBM (the version validated in the blend)')
add('04_blend_M33_T11_20_T9_17_C30.csv',0.33*M+0.2*T+0.17*T9+0.3*C,note='second pipeline = average of v11 and v9')
add('05_blend_M50_T50.csv',0.5*M+0.5*T,note='two feature pipelines only, no CNN (clean val 0.931)')
add('06_blend_M30_T30_C40.csv',0.3*M+0.3*T+0.4*C,note='heavier CNN (CNN optimum was higher on PKS-like data)')
add('07_blend_M175_Mpl175_T35_C30.csv',0.175*M+0.175*Mpl+0.35*T+0.3*C,note='mix of plain and pseudo-label HGB')
A=lab(0.35*M+0.35*T+0.3*C); vote=[]
for a,b,c in zip(A,v11.label.values,v2.label.values): vote.append(b if b==c else a)
add('08_vote_blend_v11_v2.csv',labels=np.array(vote),note='v11 & v2 labels where they agree, else main blend')
add('09_v11_theirs.csv',labels=v11.label.values,note='other pipeline v11 as-is (not LB-scored)')
add('10_v2_theirs_0.8571.csv',labels=v2.label.values,note='LB 0.8571 baseline (safety)')
pd.DataFrame(files).to_csv(f'{O}/index.csv',index=False); print(pd.DataFrame(files).to_string()); print('CNNs used',cnn_files)
