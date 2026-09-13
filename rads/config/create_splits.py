import os
import glob
import pandas as pd
import random

def extract_source_id(filename):
    # E.g. jD8ybdMZOU8_00.mp4 -> jD8ybdMZOU8
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    return base.split('.')[0]

def create_p02_test_split():
    print("Creating p02_test_split.csv...")
    p02_meta = r"e:\Rads\Datasets\processed\picek\p02_binary\picek_p02_binary_metadata.csv"
    df = pd.read_csv(p02_meta)
    
    test_df = df[df['split_in_distribution'] == 'test'].copy()
    print(f"Found {len(test_df)} test samples in P02.")
    
    out_path = r"e:\Rads\rads\config\p02_test_split.csv"
    test_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}\n")

def create_heldout_500_split():
    print("Creating picek_500_split.csv from held-out PICEK trimmed sorted dataset...")
    pos_dir = r"e:\Rads\Datasets\processed\picek_sorted\trimmed\positive\real"
    neg_dir = r"e:\Rads\Datasets\processed\picek_sorted\trimmed\negative\real"
    
    pos_files = sorted(glob.glob(os.path.join(pos_dir, "*.mp4")))
    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.mp4")))
    
    # Skip the first 30 to exclude videos used during Phase 1-7 tuning
    pos_pool = pos_files[30:]
    neg_pool = neg_files[30:]
    
    # We want strict source-video separation, so we pick unique sources
    def select_unique_sources(file_list, n):
        selected = []
        seen_sources = set()
        
        # shuffle with a seed for reproducibility
        random.seed(42)
        random.shuffle(file_list)
        
        for f in file_list:
            src = extract_source_id(f)
            if src not in seen_sources:
                seen_sources.add(src)
                selected.append(f)
            if len(selected) == n:
                break
        return selected
    
    pos_selected = select_unique_sources(pos_pool, 250)
    neg_selected = select_unique_sources(neg_pool, 250)
    
    if len(pos_selected) < 250 or len(neg_selected) < 250:
        print(f"WARNING: Could not find 250 unique sources! (Pos: {len(pos_selected)}, Neg: {len(neg_selected)})")
        print("Falling back to allowing multiple clips per source.")
        # Fallback if needed
        pos_selected = pos_pool[:250]
        neg_selected = neg_pool[:250]
    
    # Create the dataframe
    records = []
    for f in pos_selected:
        records.append({
            "video_id": os.path.basename(f),
            "binary_label": 1,
            "processed_path": f,
            "split_in_distribution": "test"
        })
        
    for f in neg_selected:
        records.append({
            "video_id": os.path.basename(f),
            "binary_label": 0,
            "processed_path": f,
            "split_in_distribution": "test"
        })
        
    df = pd.DataFrame(records)
    
    out_path = r"e:\Rads\rads\config\picek_500_split.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} records.")

if __name__ == "__main__":
    create_p02_test_split()
    create_heldout_500_split()
