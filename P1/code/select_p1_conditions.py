#!/usr/bin/env python3
"""Select measured, nonzero approximately-half-grasp cases; never invent a result."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('scan_directory', type=Path)
p.add_argument('output_json', type=Path)
a = p.parse_args()
s = json.loads((a.scan_directory / 'summary.json').read_text())
m = json.loads((a.scan_directory / 'metadata.json').read_text())
assert s['status'].startswith('METRICS'), f"Incomplete scan: {s['status']}"
rows = s['conditions']
baseline = next(r for r in rows if r['name'] == 'baseline')
assert baseline['episodes'] == 10, 'Course experiment requires 10 episodes per condition'
assert baseline['grasps'] > 0, 'Baseline has zero grasps; half-degradation cannot be defined'

def position_ood(r):
    return any(r.get(k) is not None and not lo <= r[k] <= hi for k, (lo, hi) in (
        ('cube_x', m['training_box_x_range_m']), ('cube_y', m['training_box_y_range_m'])))

selected = []
for name, kind in [('exp2_far', 'position'), ('exp3_noise', 'noise')]:
    candidates = [r for r in rows if r['name'] != 'baseline' and r['episodes'] == 10
                  and r['paired_comparison_to_baseline']['nonzero_approximately_half_grasps']
                  and ((r['obs_noise'] > 0) if kind == 'noise' else position_ood(r))]
    if not candidates:
        print(json.dumps({'status':'REFINEMENT_NEEDED', 'factor':kind,
              'baseline_grasps':baseline['grasps'], 'target_grasps':baseline['grasps']/2,
              'measured_trials':[{k:r[k] for k in ('name','cube_x','cube_y','obs_noise','grasps')} for r in rows]}, indent=2))
        raise SystemExit(2)
    chosen = min(candidates, key=lambda r: (abs(r['grasps'] - baseline['grasps']/2), r['obs_noise'], r['name']))
    selected.append({'name':name, **{k:chosen[k] for k in ('cube_x','cube_y','obs_noise')}})
    print('SELECTED', name, chosen['name'], f"{chosen['grasps']}/10 grasps", json.dumps(selected[-1]))
assert not a.output_json.exists(), 'Do not overwrite a previous selection silently'
a.output_json.write_text(json.dumps(selected, indent=2) + '\n')
print('RECORD_THESE_CONDITIONS', a.output_json)
