import os
import json
from collections import defaultdict

src_dir = '../../pairUAV/test'
dst_dir = 'test_pairs'
GROUP_SIZE = 5000

os.makedirs(dst_dir, exist_ok=True)

# Read all JSON files and collect (image_a, image_b) pairs
pairs = defaultdict(set)

for subdir in sorted(os.listdir(src_dir)):
    subdir_path = os.path.join(src_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for fname in os.listdir(subdir_path):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(subdir_path, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
        image_a = data['image_a']
        image_b = data['image_b']
        pairs[image_a].add(image_b)

# Flatten all pairs into a sorted list
all_pairs = []
for image_a, image_b_set in sorted(pairs.items()):
    name_a = os.path.splitext(image_a)[0] + '.webp'
    for b in sorted(image_b_set):
        all_pairs.append((name_a, b))

print(f'Total unique image pairs: {len(all_pairs)}')

# Group every GROUP_SIZE pairs, write to numbered txt files, and build index json
pair_to_group = {}
num_groups = (len(all_pairs) + GROUP_SIZE - 1) // GROUP_SIZE

for group_id in range(num_groups):
    start = group_id * GROUP_SIZE
    end = min(start + GROUP_SIZE, len(all_pairs))
    out_path = os.path.join(dst_dir, f'{group_id:04d}.txt')
    with open(out_path, 'w') as f:
        for i in range(start, end):
            a, b = all_pairs[i]
            f.write(f'{a} {b}\n')
            pair_to_group[f'{a} {b}'] = f'{group_id:04d}'

# Write index json: pair -> group_id
index_path = os.path.join(dst_dir, 'pair_groups.json')
with open(index_path, 'w') as f:
    json.dump(pair_to_group, f, indent=2)

print(f'Done. Generated {num_groups} group txt files and pair_groups.json in {dst_dir}')
