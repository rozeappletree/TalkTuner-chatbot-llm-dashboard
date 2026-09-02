#!/usr/bin/env bash
# Unzips every zip archive in data/dataset/ into a sibling folder of the same
# name (minus .zip). Uses python3's zipfile module since `unzip` may not be
# installed. The extracted folders are already covered by data/dataset/*/ in
# .gitignore.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$REPO_ROOT/data/dataset"

python3 - "$DATASET_DIR" <<'EOF'
import sys, zipfile, os, glob

dataset_dir = sys.argv[1]
os.chdir(dataset_dir)

for f in sorted(glob.glob("*.zip")):
    dirname = f[:-4]
    os.makedirs(dirname, exist_ok=True)
    with zipfile.ZipFile(f) as z:
        z.extractall(dirname)
    print(f"unzipped {f} -> {dirname}")
EOF
