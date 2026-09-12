import json

d = json.load(open(r'e:\Rads\rads\evaluation\experiments\frame_skip_comparison\forensics\FP_-6SQSDj8cYU_00.json'))
print('=== FP1: -6SQSDj8cYU_00 ===')
print('Max frame in video: 60')
for tid_str, ts in d['tracking']['track_summaries'].items():
    cn = ts['class_name']
    ff = ts['first_frame']
    lf = ts['last_frame']
    fc = ts['frame_count']
    print(f'  Track {tid_str}: {cn} frames {ff}-{lf} ({fc} detections)')

print()

d2 = json.load(open(r'e:\Rads\rads\evaluation\experiments\frame_skip_comparison\forensics\FP_-dmYsQc-odI_00.json'))
print('=== FP2: -dmYsQc-odI_00 ===')
total = d2['video_meta']['total_frames']
print(f'Total frames: {total}')
for tid_str, ts in d2['tracking']['track_summaries'].items():
    cn = ts['class_name']
    ff = ts['first_frame']
    lf = ts['last_frame']
    fc = ts['frame_count']
    print(f'  Track {tid_str}: {cn} frames {ff}-{lf} ({fc} detections)')

print()

# Also check FN1 pairwise closely: normalized_proximity values
d3 = json.load(open(r'e:\Rads\rads\evaluation\experiments\frame_skip_comparison\forensics\FN_-2UPLUV7JLg_00.json'))
print('=== FN1: -2UPLUV7JLg_00 ===')
print(f'Total frames: {d3["video_meta"]["total_frames"]}, FPS: {d3["video_meta"]["fps"]}')
print(f'Total unique tracks: {d3["tracking"]["total_unique_tracks"]}')
for tid_str, ts in d3['tracking']['track_summaries'].items():
    cn = ts['class_name']
    ff = ts['first_frame']
    lf = ts['last_frame']
    fc = ts['frame_count']
    print(f'  Track {tid_str}: {cn} frames {ff}-{lf} ({fc} detections)')

print()
print('FN1 - Looking for close pairs (norm_prox < 1.5):')
for pair_key, pair_data in d3['pairwise_detail'].items():
    for frame_data in pair_data['frames']:
        if frame_data['norm_prox'] < 1.5:
            print(f'  Pair {pair_key} frame {frame_data["frame"]}: norm_prox={frame_data["norm_prox"]}, iou={frame_data["iou"]}')
