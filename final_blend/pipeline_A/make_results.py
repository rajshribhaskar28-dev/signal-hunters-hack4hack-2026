import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from cv import *
E=[ # (id, file, model, features, train data, eval set, runtime_s, conclusion)
 ('E0','oof_starter.npy','HGB(300it,lr.06,balanced)','starter notebook per-panel stats (35)','GBT clean','5-fold strat GBT clean',90,'baseline'),
 ('E1','oof_v1_hgb.npy','HGB(lr.06,ES,balanced)','v1 physics: drift search, global ETI track, dedispersion, texture (654)','GBT clean','5-fold strat GBT clean',494,'+0.301 over baseline; ETI & scint weak'),
 ('E2','oof_v2_hgb.npy','HGB(lr.1,ES,balanced)','v2: +ON/OFF level & patchiness, asymmetric OFF veto (751)','GBT clean','5-fold strat GBT clean',635,'+0.032; scint 0.56->0.73 kept'),
 ('E3c','oofc_v3_clean.npy','HGB(lr.1,ES,balanced)','v3: fine global drift grid 1/120, clipped line maps (758)','GBT clean','5-fold strat GBT clean',1202,'+0.005; ETI 0.68->0.71 kept'),
 ('E3a','oofa_v3_clean.npy','HGB(lr.1,ES,balanced)','v3','GBT clean','same folds, PKS-aug v1 holdout (stress)',1202,'clean-trained model collapses under shift (noise F1 0.04) -> augmentation required'),
 ('E4c','oofc_v3_mix_aug2.npy','HGB(lr.1,ES,balanced)','v3','GBT clean + PKS-aug2','5-fold strat GBT clean',1033,'no loss on clean'),
 ('E4a','oofa_v3_mix_aug2.npy','HGB(lr.1,ES,balanced)','v3','GBT clean + PKS-aug2','same folds, PKS-aug2 holdout (stress)',1033,'robust under simulated shift -> adopted'),
 ('E5c','oofc_v4_mix_aug2.npy','HGB(lr.1,ES,balanced)','v4: +global-track panel occupancy (770)','GBT clean + PKS-aug2','5-fold strat GBT clean',660,'within noise of E4'),
 ('E5a','oofa_v4_mix_aug2.npy','HGB(lr.1,ES,balanced)','v4','GBT clean + PKS-aug2','same folds, PKS-aug2 holdout (stress)',660,'within noise of E4; RFI classes +0.006-0.009'),
 ('E6c','oofc_v4_mix_aug2_drop.npy','HGB(lr.1,ES,balanced)','v4 minus absolute noise-floor stats (keep ON-OFF diffs) (641)','GBT clean + PKS-aug2','5-fold strat GBT clean',606,'same score, fewer telescope-dependent features -> adopted'),
 ('E6a','oofa_v4_mix_aug2_drop.npy','HGB(lr.1,ES,balanced)','v4 minus absolute noise-floor stats','GBT clean + PKS-aug2','same folds, PKS-aug2 holdout (stress)',606,'same score -> adopted'),
 ('E7c','ENS:ens_v3cols_c.npy+ens_v4drop_c.npy','2x HGB prob-avg (feature sets v3cols, v4drop)','v4 file','GBT clean + PKS-aug2','5-fold strat GBT clean',1250,'small consistent gain'),
 ('E7a','ENS:ens_v3cols_a.npy+ens_v4drop_a.npy','2x HGB prob-avg','v4 file','GBT clean + PKS-aug2','same folds, PKS-aug2 holdout (stress)',1250,'+0.002 on same holdout -> adopted (earlier +0.023 was a noise-realisation artifact)'),
 ('E8c','ENS:ens_v5_v3cols_c.npy+ens_v5_v4drop_c.npy','2x HGB prob-avg','v5: + precise ON-track collinearity & drift-consistency geometry (785)','GBT clean + PKS-aug2','5-fold strat GBT clean',1300,'+0.015; ETI 0.71->0.82 -> FINAL CONFIG'),
 ('E8a','ENS:ens_v5_v3cols_a.npy+ens_v5_v4drop_a.npy','2x HGB prob-avg','v5','GBT clean + PKS-aug2','same folds, PKS-aug2 holdout (stress)',1300,'+0.013; ETI 0.61->0.69 -> FINAL CONFIG'),
]
rows=[]
for e in E:
    try:
        if e[1].startswith('ENS:'): P=sum(np.load('cache/'+f) for f in e[1][4:].split('+'))/2
        else: P=np.load('cache/'+e[1])
    except FileNotFoundError: continue
    pred=P.argmax(1); pc=f1_score(y,pred,average=None)
    # fold-wise std of macro-F1
    skf=StratifiedKFold(5,shuffle=True,random_state=42); fs=[f1_score(y[b],pred[b],average='macro') for a,b in skf.split(P,y)]
    r=dict(exp=e[0],model=e[2],features=e[3],train_data=e[4],validation=e[5],cv_seed=42,
           macro_f1=round(f1_score(y,pred,average='macro'),4),fold_std=round(np.std(fs),4),runtime_s=e[6])
    r.update({f'f1_{c}':round(v,4) for c,v in zip(CLASSES,pc)}); r['conclusion']=e[7]; rows.append(r)
R=pd.DataFrame(rows); R.to_csv('experiment_results.csv',index=False); print(R[['exp','macro_f1','fold_std','f1_eti_candidate','f1_scintillating_star']].to_string())
