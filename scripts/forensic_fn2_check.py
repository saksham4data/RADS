import json

d = json.load(open(r'e:\Rads\rads\evaluation\experiments\frame_skip_comparison\forensics\FN_-9oifpjUxxM_00.json'))
print('FN2 Close pairs (norm_prox < 1.5):')
found = False
for k, v in d['pairwise_detail'].items():
    for f in v['frames']:
        if f['norm_prox'] < 1.5:
            found = True
            print(f"  Pair {k}: frame {f['frame']} norm_prox={f['norm_prox']} iou={f['iou']} rel_vel={f['rel_vel']}")

if not found:
    print('  NONE FOUND - no pair ever gets close enough')

print()
print('FN2 Closest approaches per pair:')
for k, v in d['pairwise_detail'].items():
    min_prox = min(f['norm_prox'] for f in v['frames'])
    max_iou = max(f['iou'] for f in v['frames'])
    if min_prox < 3.0:
        print(f"  Pair {k}: min_norm_prox={min_prox:.3f} max_iou={max_iou:.4f} frames={v['co_existing_frames']}")
