#!/usr/bin/env bash
# Unzips every zip archive in data/probe_checkpoints/ into a sibling folder of
# the same name (minus .zip). Uses python3's zipfile module since `unzip` may
# not be installed. The extracted folders are already covered in .gitignore.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS_DIR="$REPO_ROOT/data/probe_checkpoints"

python3 - "$CHECKPOINTS_DIR" <<'EOF'
import sys, zipfile, os, glob

checkpoints_dir = sys.argv[1]
os.chdir(checkpoints_dir)

for f in sorted(glob.glob("*.zip")):
    dirname = f[:-4]
    os.makedirs(dirname, exist_ok=True)
    with zipfile.ZipFile(f) as z:
        z.extractall(dirname)
    print(f"unzipped {f} -> {dirname}")
EOF
