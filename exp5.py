import sys, json
import final_model
from final_model import validate
final_model.SELECT = json.load(open('selected_features_350.json'))
grid = {
 'it400_lr06': dict(max_iter=400, lr=0.06, bins=64),
 'it300_lr06_l63': dict(max_iter=300, lr=0.06, bins=64, leaves=63, msl=30),
 'it400_lr06_l15': dict(max_iter=400, lr=0.06, bins=64, leaves=15),
}
for k in sys.argv[1].split(','):
    validate({'groups': '1234', 'augs': [2], 'val_aug': 2, 'hgb': grid[k]}, fold=0, name=f'sel350_{k}')
