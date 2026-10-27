import sys, pandas as pd
import os
D = os.environ.get('SH_DATA', 'data')
L = ['eti_candidate','rfi_persistent','rfi_intermittent','satellite_pass','pulsar','frb','radar_chirp','scintillating_star','noise']
s = pd.read_csv(sys.argv[1]); ss = pd.read_csv(f'{D}/sample_submission.csv'); te = pd.read_csv(f'{D}/test_meta.csv')
checks = {
 'rows == 10000': len(s) == 10000,
 'columns exactly [id,label]': list(s.columns) == ['id', 'label'],
 'ids match sample order': (s.id.values == ss.id.values).all(),
 'ids match test_meta': set(s.id) == set(te.id),
 'ids unique': s.id.is_unique,
 'no missing labels': s.label.notna().all() and (s.label.astype(str).str.len() > 0).all(),
 'labels in allowed set': s.label.isin(L).all(),
}
for k, v in checks.items(): print(f'[{"PASS" if v else "FAIL"}] {k}')
print('ALL PASS' if all(checks.values()) else 'FAILED')
