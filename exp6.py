import sys, json
import final_model
from final_model import validate
cfg = json.load(open('final_config.json')); cfg['select_in_fold'] = True
for spec in sys.argv[1:]:
    name, groups = spec.split(':')
    cfg['groups'] = groups
    validate(cfg, fold=0, name=name)
