import os
import shutil
import glob
from pathlib import Path

base_dir = Path("e:/Rads/training/outputs")
archive_dir = base_dir / "archive"

# 1. TUDAT E01-E06 (Aug 5 to Aug 13)
e01_e06_dir = archive_dir / "tudat_e01_e06"
# 2. TUDAT E07-T01 (Aug 24 to Aug 25)
e07_t01_dir = archive_dir / "tudat_e07_t01"
# 3. P01 Hyperopt (Aug 29 to Aug 30 11-21)
p01_hyp_dir = archive_dir / "p01_hyperopt_study"
# 4. P01 Duplicates (Aug 30 11-27 to 11-43)
p01_dup_dir = archive_dir / "p01_duplicate_tests"

for d in base_dir.iterdir():
    if not d.is_dir() or d.name == "archive" or d.name == "logs" or d.name == "latest":
        continue
    
    # Check date patterns
    if d.name.startswith("2026-08-0") or d.name.startswith("2026-08-1"):
        shutil.move(str(d), str(e01_e06_dir / d.name))
    elif d.name.startswith("2026-08-24") or d.name.startswith("2026-08-25"):
        shutil.move(str(d), str(e07_t01_dir / d.name))
    elif d.name.startswith("2026-08-29"):
        shutil.move(str(d), str(p01_hyp_dir / d.name))
    elif d.name.startswith("2026-08-30"):
        # Aug 30 splits
        time_part = d.name.split("_")[1] # e.g. 11-16-50
        hour_min = int(time_part[:5].replace("-", "")) # 1116
        if hour_min <= 1121:
            shutil.move(str(d), str(p01_hyp_dir / d.name))
        else:
            shutil.move(str(d), str(p01_dup_dir / d.name))

# Remove duplicate test runs completely
for d in p01_dup_dir.iterdir():
    shutil.rmtree(d)

# Clean pycache globally in training
for p in Path("e:/Rads/training").rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

print("Cleanup complete!")
