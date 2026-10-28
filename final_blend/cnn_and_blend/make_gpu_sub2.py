import sys, numpy as np, pandas as pd
CL=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
W='/home/claude/work'; V='/home/claude/night/v11/cache'; D='/home/claude/data/signal-hunters-round2'
npzs=sys.argv[1].split(','); out=sys.argv[2]; wg=float(sys.argv[3]) if len(sys.argv)>3 else 0.4
ss=pd.read_csv(f'{D}/sample_submission.csv')
M=np.load(f'{W}/cache/test_proba.npy')
T=0.8*np.load(f'{V}/testP_v10.npy')+0.2*np.mean([np.load(f'{V}/testP_map_mf{s}.npy') for s in range(3)],0)
ld=lambda p: np.mean([np.load(f'{W}/cache/{q}_test.npy') for q in p.split('+')],0)
Co=np.mean([ld('gnp_s2_h0+gnp_s3_h0_pl'),ld('np_s0_h0'),ld('cnn_s0_h0+cnn_s1_h0')],0)
gs=[np.load(n) for n in npzs]; wts=np.array([int(g['seeds_done']) for g in gs],float)
G=sum(w*g['test'] for w,g in zip(wts,gs))/wts.sum()
L=lambda P: np.log(np.clip(P,1e-6,1))
wm=(0.9-wg)/2
S=wm*L(M)+wm*L(T)+wg*L(G)+0.1*L(Co)
lab=np.array(CL)[S.argmax(1)]
s=pd.DataFrame({'id':ss.id.values,'label':lab})
assert list(s.columns)==['id','label'] and len(s)==10000 and s.id.is_unique and (s.id.values==ss.id.values).all() and s.label.isin(CL).all() and s.label.notna().all()
s.to_csv(out,index=False)
best=pd.read_csv('/home/claude/night/out/11_blend_gpu_M25_T25_G40_C10.csv')
print('wrote',out,'| GPU models',int(wts.sum()),'| agreement with 0.8872 file:',round((best.label==s.label).mean(),4))
print(s.label.value_counts().to_dict())
