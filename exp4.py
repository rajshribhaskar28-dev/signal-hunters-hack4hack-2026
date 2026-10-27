import sys, json
import final_model
from final_model import validate
if len(sys.argv) > 5: final_model.SELECT = json.load(open(sys.argv[5]))
cfg = {'groups': sys.argv[2], 'augs': [int(a) for a in sys.argv[3].split(',')], 'val_aug': 2,
       'hgb': dict(max_iter=150, lr=0.12, bins=64)}
validate(cfg, fold=int(sys.argv[4]) if len(sys.argv) > 4 else 0, name=sys.argv[1])
