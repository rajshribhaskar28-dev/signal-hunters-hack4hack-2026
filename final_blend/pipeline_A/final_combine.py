import numpy as np, pandas as pd
CL=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
D='/home/claude/data/signal-hunters-round2'
H=(np.load('cache/test_proba_agree_v4drop.npy')+np.load('cache/test_proba_agree_v3cols.npy'))/2
C=np.mean([np.load(f'cache/cnn_s{s}_h0_test.npy') for s in (0,1)],0)
P=0.65*H+0.35*C; np.save('submission_final_v4_proba.npy',P)
a=pd.read_csv('/mnt/user-data/attachments/52491fa9-e946-55d9-84b3-5f2ccd6e5936/716c67d2-submission_v2.csv'); ss=pd.read_csv(f'{D}/sample_submission.csv')
ours=np.array(CL)[P.argmax(1)]; conf=P.max(1)
d=(a.label.values!=ours)
h=a.copy(); m=d&(conf>=0.7); h.loc[m,'label']=ours[m]
ok=list(h.columns)==['id','label'] and len(h)==10000 and h.id.is_unique and (h.id.values==ss.id.values).all() and h.label.isin(CL).all() and h.label.notna().all()
assert ok
h.to_csv('submission_hybrid_v2.csv',index=False)
old=pd.read_csv('submission_hybrid.csv')
print('schema ok',ok,'| disagreements with theirs',d.sum(),'| overrides',m.sum(),'| agree w/ hybrid v1',(h.label==old.label).mean().round(4))
print(pd.Series(ours[m]).value_counts().to_dict())
