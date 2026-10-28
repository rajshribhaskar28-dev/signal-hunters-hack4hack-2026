import os
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
seeds=[int(s) for s in sys.argv[1].split(',')]
hoi=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))[0][1]
yh=y[hoi]
for d in ['c','a']:
    H=((np.load(f'cache/ens_v5_v3cols_{d}.npy')+np.load(f'cache/ens_v5_v4drop_{d}.npy'))/2)[hoi]
    C=np.mean([np.load(f'cache/cnn_s{s}_h0_hold_{d}.npy') for s in seeds],0)
    print('==',{'c':'clean','a':'PKS-aug'}[d])
    for w in [0,0.2,0.3,0.4,0.5,0.6,1.0]:
        P=(1-w)*H+w*C; pc=f1_score(yh,P.argmax(1),average=None)
        print(f' w_cnn={w:.1f} macro={pc.mean():.4f} ',' '.join(f'{c[:5]}={v:.3f}' for c,v in zip(CLASSES,pc)))
