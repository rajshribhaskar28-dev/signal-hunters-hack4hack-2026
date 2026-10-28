import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
CLASSES=['eti_candidate','frb','pulsar','scintillating_star','satellite_pass','radar_chirp','rfi_intermittent','rfi_persistent','noise']
D=os.environ.get('SH_DATA','data')
seeds=[int(s) for s in sys.argv[1].split(',')]; W=float(sys.argv[2]); out=sys.argv[3]
H=np.load('cache/test_proba.npy'); C=np.mean([np.load(f'cache/cnn_s{s}_h0_test.npy') for s in seeds],0)
P=(1-W)*H+W*C; np.save(out.replace('.csv','_proba.npy'),P)
te=pd.read_csv(f'{D}/test_meta.csv'); ss=pd.read_csv(f'{D}/sample_submission.csv')
sub=pd.DataFrame({'id':te.id.values,'label':np.array(CLASSES)[P.argmax(1)]})
assert list(sub.columns)==['id','label'] and len(sub)==10000 and sub.id.is_unique and (sub.id.values==ss.id.values).all()
assert sub.label.isin(CLASSES).all() and sub.label.notna().all()
sub.to_csv(out,index=False)
prev=pd.read_csv('submission_final.csv')
print('wrote',out,'agreement with submission_final:',round((prev.label==sub.label).mean(),4)); print((sub.label.value_counts(normalize=True)*100).round(1).to_dict())
