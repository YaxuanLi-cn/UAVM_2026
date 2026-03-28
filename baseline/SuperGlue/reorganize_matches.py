#!/usr/bin/env python3
"""Reorganize matches data from origin_matches_data/{NNNN}/{id1}_{id2}_matches.npz
to test_matches_data/{id1}/{id2}.npz"""

import os
import shutil
from pathlib import Path

src_root = Path("./origin_test_matches_data")
dst_root = Path("../test_matches_data")

count = 0
errors = 0

for bucket_dir in sorted(src_root.iterdir()):
    if not bucket_dir.is_dir():
        continue
    for npz_file in bucket_dir.iterdir():
        if not npz_file.name.endswith("_matches.npz"):
            continue
        # filename: {id1}_{id2}_matches.npz
        stem = npz_file.name[:-len("_matches.npz")]  # remove _matches.npz
        # IDs are 12 chars each, separated by _
        if len(stem) != 25 or stem[12] != '_':
            print(f"WARNING: unexpected filename format: {npz_file.name}")
            errors += 1
            exit(0)
        id1 = stem[:12]
        id2 = stem[13:25]

        dst_dir = dst_root / id1
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / f"{id2}.npz"
        shutil.copy2(str(npz_file), str(dst_file))
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} files...")

print(f"Done. Copied {count} files, {errors} errors.")
