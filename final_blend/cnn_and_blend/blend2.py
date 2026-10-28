import sys; sys.path.insert(0,'/home/claude/work')
from cv import *
from itertools import product
hoi=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))[0][1]; yh=y[hoi]
groups=[g.split('+') for g in sys.argv[1:]]   # each arg: '+'-joined prefixes averaged into one CNN component
def load(prefixes,d): return np.mean([np.load(f'cache/{p}_hold_{d}.npy') for p in prefixes],0)
for d in ['c','a']:
    H=((np.load(f'cache/ens_v5_v3cols_{d}.npy')+np.load(f'cache/ens_v5_v4drop_{d}.npy'))/2)[hoi]
    Cs=[load(g,d) for g in groups]
    print('==',{'c':'clean','a':'PKS-aug'}[d],'HGB alone %.4f'%f1_score(yh,H.argmax(1),average='macro'),
          ' CNNs alone:',[round(f1_score(yh,C.argmax(1),average='macro'),4) for C in Cs])
    best=[]
    for w in [0.2,0.3,0.35,0.4,0.5]:
        C=np.mean(Cs,0); P=(1-w)*H+w*C; best.append((f1_score(yh,P.argmax(1),average='macro'),w))
    print('  HGB+meanCNN by w:',[(w,round(s,4)) for s,w in best])
