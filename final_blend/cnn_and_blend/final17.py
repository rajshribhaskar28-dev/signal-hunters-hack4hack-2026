import sys, numpy as np, pandas as pd
CL=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
W='/home/claude/work'; V='/home/claude/night/v11/cache'; D='/home/claude/data/signal-hunters-round2'; G_='/home/claude/night/gpu'
ss=pd.read_csv(f'{D}/sample_submission.csv')
M=np.load(f'{W}/cache/test_proba.npy')
T=0.8*np.load(f'{V}/testP_v10.npy')+0.2*np.mean([np.load(f'{V}/testP_map_mf{s}.npy') for s in range(3)],0)
ld=lambda p: np.mean([np.load(f'{W}/cache/{q}_test.npy') for q in p.split('+')],0)
Co=np.mean([ld('gnp_s2_h0+gnp_s3_h0_pl'),ld('np_s0_h0'),ld('cnn_s0_h0+cnn_s1_h0')],0)
V1=np.load(f'{G_}/V1_test.npy'); a=np.load(f'{G_}/in/gpuv2.npz'); b=np.load(sys.argv[1])
V2=(int(a['seeds_done'])*a['test']+int(b['seeds_done'])*b['test'])/(int(a['seeds_done'])+int(b['seeds_done']))
G=0.33*V1+0.67*V2
L=lambda P: np.log(np.clip(P,1e-6,1))
wg=0.5; wm=(0.9-wg)/2; S=wm*L(M)+wm*L(T)+wg*L(G)+0.1*L(Co)
s=pd.DataFrame({'id':ss.id.values,'label':np.array(CL)[S.argmax(1)]})
assert list(s.columns)==['id','label'] and len(s)==10000 and s.id.is_unique and (s.id.values==ss.id.values).all() and s.label.isin(CL).all() and s.label.notna().all()
out='/home/claude/night/out/17_final_v2x4_G50.csv'; s.to_csv(out,index=False)
p=pd.read_csv('/home/claude/night/out/15_logblend_v2heavy_G50.csv')
print('wrote',out,'v2 seeds',int(a['seeds_done'])+int(b['seeds_done']),'| agreement with 0.8906 file',round((p.label==s.label).mean(),4)); print(s.label.value_counts().to_dict())
